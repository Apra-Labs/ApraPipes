#include "H265Decoder.h"

#ifdef ARM64
#include "H264DecoderV4L2Helper.h"
#else
#include "H264DecoderNvCodecHelper.h"
#endif

// #include "H265ParserUtils.h"  // Not needed for current implementation
#include "FrameMetadata.h"
#include "H265Metadata.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "H265Utils.h"
#include <linux/videodev2.h>
#ifdef ARM64
#include "v4l2_nv_extensions.h"
#endif
#include <deque>
#include <mutex>

class H265Decoder::Detail
{
public:
	Detail(H265DecoderProps& _props) : mWidth(0), mHeight(0)
	{
	}

	~Detail()
	{
		helper.reset();
	}

	bool setMetadata(framemetadata_sp& metadata, frame_sp frame, std::function<void(frame_sp&)> send, std::function<frame_sp()> makeFrame)
	{
		auto type = H265Utils::getNALUType((char*)frame->data());
		if (H265Utils::isIDR(type) || type == H265Utils::H265_NAL_TYPE::VPS || type == H265Utils::H265_NAL_TYPE::SPS)
		{
			if (metadata->getFrameType() == FrameMetadata::FrameType::HEVC_DATA)
			{
				// For now, use basic dimensions - would need H265ParserUtils for proper SPS parsing
				mWidth = 1920;  // Default width - should parse from SPS
				mHeight = 1080; // Default height - should parse from SPS

				auto h265Metadata = framemetadata_sp(new H265Metadata(mWidth, mHeight));
				auto rawOutMetadata = FrameMetadataFactory::downcast<H265Metadata>(h265Metadata);
				rawOutMetadata->setData(*rawOutMetadata);
#ifdef ARM64
				helper.reset(new h264DecoderV4L2Helper());
				return helper->init(send, makeFrame, V4L2_PIX_FMT_H265);
#else
				helper.reset(new H264DecoderNvCodecHelper(mWidth, mHeight));
				return helper->init(send, makeFrame);
#endif
			}
			else
			{
				throw AIPException(AIP_NOTIMPLEMENTED, "Unknown frame type");
			}
		}
		else
		{
			return false;
		}
	}

	void compute(void* inputFrameBuffer, size_t inputFrameSize, uint64_t inputFrameTS)
	{
		if(helper != nullptr)
		{
			helper->process(inputFrameBuffer, inputFrameSize, inputFrameTS);
		}
	}

#ifdef ARM64
	void closeAllThreads(frame_sp eosFrame)
	{
		helper->closeAllThreads(eosFrame);
	}
#endif
public:
	int mWidth;
	int mHeight;
private:

#ifdef ARM64
	boost::shared_ptr<h264DecoderV4L2Helper> helper;
#else
	boost::shared_ptr<H264DecoderNvCodecHelper> helper;
#endif
};

H265Decoder::H265Decoder(H265DecoderProps _props) : Module(TRANSFORM, "H265Decoder", _props), mShouldTriggerSOS(true), mProps(_props)
{
	mDetail.reset(new Detail(mProps));
#ifdef ARM64
	mOutputMetadata = boost::shared_ptr<FrameMetadata>(new RawImageMetadata(FrameMetadata::MemType::DMABUF));
#else
	mOutputMetadata = boost::shared_ptr<FrameMetadata>(new RawImagePlanarMetadata(RawImageMetadata::MemType::HOST));
#endif
	mOutputPinId = Module::addOutputPin(mOutputMetadata);
}

H265Decoder::~H265Decoder() {}

bool H265Decoder::init()
{
	if (!Module::init())
	{
		return false;
	}

	return true;
}

bool H265Decoder::term()
{
#ifdef ARM64
	auto eosFrame = frame_sp(new EoSFrame());
	mDetail->closeAllThreads(eosFrame);
#endif
	mDetail.reset();
	return Module::term();
}

void* H265Decoder::prependVpsSpsPps(frame_sp& iFrame, size_t& vpsSpsPpsFrameSize)
{
	// Calculate total size: original frame + VPS + SPS + PPS + 4-byte NAL separators
	size_t vpsSize = vpsBuffer.size();
	size_t spsSize = spsBuffer.size();
	size_t ppsSize = ppsBuffer.size();
	size_t totalHeaderSize = vpsSize + spsSize + ppsSize;
	size_t numSeparators = 0;

	// Count non-empty headers for NAL separators
	if (vpsSize > 0) numSeparators++;
	if (spsSize > 0) numSeparators++;
	if (ppsSize > 0) numSeparators++;

	vpsSpsPpsFrameSize = iFrame->size() + totalHeaderSize + (numSeparators * 4);
	uint8_t* vpsSpsPpsFrameBuffer = new uint8_t[vpsSpsPpsFrameSize];
	char NaluSeparator[4] = { 0x00, 0x00, 0x00, 0x01 };
	auto nalu = reinterpret_cast<uint8_t*>(NaluSeparator);

	uint8_t* bufferPtr = vpsSpsPpsFrameBuffer;

	// Prepend VPS if present
	if (vpsSize > 0)
	{
		memcpy(bufferPtr, nalu, 4);
		bufferPtr += 4;
		memcpy(bufferPtr, vpsBuffer.data(), vpsSize);
		bufferPtr += vpsSize;
	}

	// Prepend SPS if present
	if (spsSize > 0)
	{
		memcpy(bufferPtr, nalu, 4);
		bufferPtr += 4;
		memcpy(bufferPtr, spsBuffer.data(), spsSize);
		bufferPtr += spsSize;
	}

	// Prepend PPS if present
	if (ppsSize > 0)
	{
		memcpy(bufferPtr, nalu, 4);
		bufferPtr += 4;
		memcpy(bufferPtr, ppsBuffer.data(), ppsSize);
		bufferPtr += ppsSize;
	}

	// Copy original frame data
	memcpy(bufferPtr, iFrame->data(), iFrame->size());

	return vpsSpsPpsFrameBuffer;
}

void H265Decoder::saveVpsSpsPps(frame_sp frame)
{
	auto mFrameBuffer = const_buffer(frame->data(), frame->size());
	auto ret = H265Utils::parseNalu(mFrameBuffer);
	const_buffer tempVpsBuffer;
	const_buffer tempSpsBuffer;
	const_buffer tempPpsBuffer;
	short typeFound;
	tie(typeFound, tempVpsBuffer, tempSpsBuffer, tempPpsBuffer) = ret;

	if ((tempVpsBuffer.size() != 0) || (tempSpsBuffer.size() != 0) || (tempPpsBuffer.size() != 0))
	{
		mHeaderFrame = frame;
		vpsBuffer = tempVpsBuffer;
		spsBuffer = tempSpsBuffer;
		ppsBuffer = tempPpsBuffer;
	}
}

bool H265Decoder::validateInputPins()
{
	if (getNumberOfInputPins() != 1)
	{
		LOG_ERROR << "This module supports only one input pin.";
		return false;
	}

	framemetadata_sp metadata = getFirstInputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::FrameType::HEVC_DATA)
	{
		LOG_ERROR << "Input pin should be HEVC_DATA. Actual type = " << frameType;
		return false;
	}

	return true;
}

bool H265Decoder::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{
		LOG_ERROR << "This module supports only one output pin.";
		return false;
	}

	return true;
}

void H265Decoder::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);
}

bool H265Decoder::processEOS(string& pinId)
{
	auto eosFrame = frame_sp(new EoSFrame());
	mDetail->closeAllThreads(eosFrame);
	return true;
}

bool H265Decoder::shouldTriggerSOS()
{
	return mShouldTriggerSOS;
}

void H265Decoder::flushQue()
{
	// Implementation for flushing queues
}

bool H265Decoder::handleCommand(Command::CommandType type, frame_sp& frame)
{
	// Implementation for handling commands
	return true;
}

bool H265Decoder::processSOS(frame_sp& frame)
{
	auto metadata = frame->getMetadata();
	auto ret = mDetail->setMetadata(metadata, frame, [&](frame_sp& outputFrame) {
		frame_container frames;
		frames.insert(make_pair(mOutputPinId, outputFrame));
		Module::send(frames);
	}, [&]() -> frame_sp {
		return makeFrame();
	});
	if (ret)
	{
		mShouldTriggerSOS = false;
#ifdef ARM64
		auto rawOutMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata);
		RawImageMetadata OutputMetadata(mDetail->mWidth, mDetail->mHeight, ImageMetadata::ImageType::RGBA, CV_8UC4, size_t(0), CV_8U, FrameMetadata::MemType::DMABUF, true);
		rawOutMetadata->setData(OutputMetadata);
#else
		auto rawOutMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(mOutputMetadata);
		RawImagePlanarMetadata OutputMetadata(mDetail->mWidth, mDetail->mHeight, ImageMetadata::YUV420, size_t(0), CV_8U, FrameMetadata::HOST);
		rawOutMetadata->setData(OutputMetadata);
#endif
	}
	return ret;
}

bool H265Decoder::process(frame_container& frames)
{
	if(incomingFramesTSQ.size() >= 1000)
	{
		flushQue();
	}
	auto frame = frames.begin()->second;
	auto myId = Module::getId();
	auto frameMetadata = frame->getMetadata();
	auto h265Metadata = FrameMetadataFactory::downcast<H265Metadata>(frameMetadata);

	// Get NAL type and check if IDR
	auto nalType = H265Utils::getNALUType((char*)frame->data());

	// Save VPS/SPS/PPS headers when encountered
	if (nalType == H265Utils::VPS || nalType == H265Utils::SPS || nalType == H265Utils::PPS)
	{
		saveVpsSpsPps(frame);
	}

	// If this is an IDR frame, prepend VPS+SPS+PPS headers
	if (H265Utils::isIDR(nalType))
	{
		size_t vpsSpsPpsFrameSize;
		auto vpsSpsPpsFrameBuffer = prependVpsSpsPps(frame, vpsSpsPpsFrameSize);
		mDetail->compute(vpsSpsPpsFrameBuffer, vpsSpsPpsFrameSize, frame->timestamp);
		delete[] static_cast<uint8_t*>(vpsSpsPpsFrameBuffer);
	}
	else
	{
		// For non-IDR frames, send as-is
		mDetail->compute(frame->data(), frame->size(), frame->timestamp);
	}

	return true;
}

void H265Decoder::bufferDecodedFrames(frame_sp& frame)
{
	// Implementation for buffering decoded frames
}

void H265Decoder::bufferBackwardEncodedFrames(frame_sp& frame, short naluType)
{
	// Implementation for buffering backward encoded frames
}

void H265Decoder::bufferAndDecodeForwardEncodedFrames(frame_sp& frame, short naluType)
{
	// Implementation for buffering and decoding forward encoded frames
}

void H265Decoder::sendDecodedFrame()
{
	// Implementation for sending decoded frames
}

void H265Decoder::decodeFrameFromBwdGOP()
{
	// Implementation for decoding frames from backward GOP
}

void H265Decoder::clearIncompleteBwdGopTsFromIncomingTSQ(std::deque<frame_sp>& latestGop)
{
	// Implementation for clearing incomplete backward GOP timestamps
}

void H265Decoder::dropFarthestFromCurrentTs(uint64_t ts)
{
	// Implementation for dropping frames farthest from current timestamp
}