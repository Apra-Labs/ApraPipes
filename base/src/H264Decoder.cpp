#include "H264Decoder.h"

#ifdef ARM64
#include "H264DecoderV4L2Helper.h"
#else
#include "H264DecoderNvCodecHelper.h"
#endif 

#include "H264ParserUtils.h"
#include "FrameMetadata.h"
#include "H264Metadata.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "H264Utils.h"

class H264Decoder::Detail
{
public:
	Detail(H264DecoderProps& _props) : mWidth(0), mHeight(0)
	{
	}

	~Detail()
	{
		helper.reset();
	}

	bool setMetadata(framemetadata_sp& metadata, frame_sp frame, std::function<void(frame_sp&)> send, std::function<frame_sp()> makeFrame)
	{
		auto type = H264Utils::getNALUType((char*)frame->data());
		if (type == H264Utils::H264_NAL_TYPE_IDR_SLICE || type == H264Utils::H264_NAL_TYPE_SEQ_PARAM )
		{
			if (metadata->getFrameType() == FrameMetadata::FrameType::H264_DATA)
			{
				sps_pps_properties p;
				H264ParserUtils::parse_sps(((const char*)frame->data()) + 5, frame->size() > 5 ? frame->size() - 5 : frame->size(), &p);
				mWidth = p.width;
				mHeight = p.height;

				auto h264Metadata = framemetadata_sp(new H264Metadata(mWidth, mHeight));
				auto rawOutMetadata = FrameMetadataFactory::downcast<H264Metadata>(h264Metadata);
				rawOutMetadata->setData(*rawOutMetadata);
				#ifdef ARM64
					helper.reset(new h264DecoderV4L2Helper());
				return helper->init(send, makeFrame);
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

	void compute(frame_sp& frame)
	{
		helper->process(frame);
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

H264Decoder::H264Decoder(H264DecoderProps _props) : Module(TRANSFORM, "H264Decoder", _props), mShouldTriggerSOS(true), mProps(_props)
{
	mDetail.reset(new Detail(mProps));
#ifdef ARM64
	mOutputMetadata = boost::shared_ptr<FrameMetadata>(new RawImagePlanarMetadata(FrameMetadata::MemType::DMABUF));
#else
	mOutputMetadata = boost::shared_ptr<FrameMetadata>(new RawImagePlanarMetadata(RawImageMetadata::MemType::HOST));
#endif
	mOutputPinId = Module::addOutputPin(mOutputMetadata);
}

H264Decoder::~H264Decoder() {}

bool H264Decoder::validateInputPins()
{
	auto numberOfInputPins = getNumberOfInputPins();
	if (numberOfInputPins != 1 && numberOfInputPins != 2)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1 or 2. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstInputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::FrameType::H264_DATA)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be H264_DATA. Actual<" << frameType << ">";
		return false;
	}

	return true;
}

bool H264Decoder::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstOutputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::RAW_IMAGE_PLANAR)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input frameType is expected to be RAW_IMAGE_PLANAR. Actual<" << frameType << ">";
		return false;
	}

	return true;
}

void H264Decoder::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);
}

bool H264Decoder::init()
{
	if (!Module::init())
	{
		return false;
	}

	return true;
}

bool H264Decoder::term()
{
#ifdef ARM64
	auto eosFrame = frame_sp(new EoSFrame());
	mDetail->closeAllThreads(eosFrame);
#endif

	mDetail.reset();
	return Module::term();
}

bool H264Decoder::checkFrameDirection(frame_sp& frame)
{
	auto frameMetadata = frame->getMetadata();
	auto h264Metadata = FrameMetadataFactory::downcast<H264Metadata>(frameMetadata);
	direction = h264Metadata->direction;
	if (!h264Metadata->direction)
	{
		tempGop.push_back(frame);
		frameCount++;
		short naluType = H264Utils::getNALUType((char*)frame->data());
		if (naluType == H264Utils::H264_NAL_TYPE_IDR_SLICE || naluType == H264Utils::H264_NAL_TYPE_SEQ_PARAM)
		{
			foundReverseGopIFrame = true;
			framesInGopCount.push(frameCount);
			gop.push_back(tempGop);
			tempGop.clear();
			frameCount = 0;
		}
		return false;
	}
	return true;
}

bool H264Decoder::process(frame_container& frames)
{
	auto frame = frames.cbegin()->second;
	auto ret = checkFrameDirection(frame);
	
	if (ret)
	{
		mDetail->compute(frame);
	}
	else if(foundReverseGopIFrame)
	{
		if (!gop.front().empty())
		{
			for (auto itr = gop.front().rbegin(); itr != gop.front().rend();)
			{
				short naluType = H264Utils::getNALUType((char*)itr->get()->data());
				if (naluType != H264Utils::H264_NAL_TYPE_IDR_SLICE || naluType != H264Utils::H264_NAL_TYPE_SEQ_PARAM || ((naluType == H264Utils::H264_NAL_TYPE_IDR_SLICE || naluType == H264Utils::H264_NAL_TYPE_SEQ_PARAM) && framesInGopCount.front() == 1))
				{
					mDetail->compute(*itr);
					itr = decltype(itr){gop.front().erase(std::next(itr).base())};
				}
				else
				{
					foundReverseGopIFrame = false;
					break;
				}
			}
		}

		if (gop.front().empty())
		{
			gop.pop_front();
			foundReverseGopIFrame = false;
		}
	}
	sendDecodedFrame();

	return true;
}

void H264Decoder::sendDecodedFrame()
{
	if (decodedFrames.size())
	{
		auto outFrame = decodedFrames.front().rbegin();
		frame_container frames;
		frames.insert(make_pair(mOutputPinId, *outFrame));
		send(frames);
		decodedFrames.front().pop_back();
	}

	if (!decodedFrames.empty())
	{
		if (decodedFrames.front().empty())
		{
			decodedFrames.pop_front();
		}
	}
}

void H264Decoder::bufferDecodedFrame(frame_sp& frame)
{
	if (!direction)
	{
		tempDecodedFrames.push_back(frame);
		auto fNoInGop = framesInGopCount.front();
		if (tempDecodedFrames.size() >= framesInGopCount.front())
		{
			decodedFrames.push_back(tempDecodedFrames);
			framesInGopCount.pop();
			tempDecodedFrames.clear();
		}
	}
	else
	{
		frame_container frames;
		frames.insert(make_pair(mOutputPinId, frame));
		send(frames);
	}
}

bool H264Decoder::processSOS(frame_sp& frame)
{
	auto metadata = frame->getMetadata();
	auto ret = mDetail->setMetadata(metadata, frame,
		[&](frame_sp& outputFrame) {
			bufferDecodedFrame(outputFrame);
		}, [&]() -> frame_sp {return makeFrame(); }
		);
	if (ret)
	{
		mShouldTriggerSOS = false;
		auto rawOutMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(mOutputMetadata);

#ifdef ARM64
		RawImagePlanarMetadata OutputMetadata(mDetail->mWidth, mDetail->mHeight, ImageMetadata::ImageType::NV12, 128, CV_8U, FrameMetadata::MemType::DMABUF);
#else
		RawImagePlanarMetadata OutputMetadata(mDetail->mWidth, mDetail->mHeight, ImageMetadata::YUV420, size_t(0), CV_8U, FrameMetadata::HOST);
#endif

		rawOutMetadata->setData(OutputMetadata);
	}
	
	return true;
}

bool H264Decoder::shouldTriggerSOS()
{
	return mShouldTriggerSOS;
}

bool H264Decoder::processEOS(string& pinId)
{
	auto frame = frame_sp(new EmptyFrame());
	mDetail->compute(frame);
	mShouldTriggerSOS = true;
	return true;
}

//Todo : override flusQue method here