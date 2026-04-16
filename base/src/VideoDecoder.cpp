#include "VideoDecoder.h"

#ifdef ARM64
#include "H264DecoderV4L2Helper.h"
#else
#include "H264DecoderNvCodecHelper.h"
#endif

#include "H264ParserUtils.h"
#include "H264Utils.h"
#include "H265Utils.h"
#include "FrameMetadata.h"
#include "H264Metadata.h"
#include "H265Metadata.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include <linux/videodev2.h>
#include <deque>
#include <mutex>
#ifdef ARM64
#include "v4l2_nv_extensions.h"
#endif

class VideoDecoder::Detail
{
public:
	Detail() : mWidth(0), mHeight(0), mCodecPixFmt(0), mNeedsHelperInit(false)
	{
	}

	~Detail()
	{
		helper.reset();
	}

	/* Buffer a decoded frame from the capture thread (thread-safe). */
	void bufferDecodedFrame(frame_sp& frame)
	{
		std::lock_guard<std::mutex> lock(mFramesMutex);
		mDecodedFrames.push_back(frame);
	}

	/* Drain all buffered frames (called from module thread). */
	std::deque<frame_sp> drainDecodedFrames()
	{
		std::lock_guard<std::mutex> lock(mFramesMutex);
		std::deque<frame_sp> out;
		out.swap(mDecodedFrames);
		return out;
	}

	/*
	 * Parse the metadata and remember codec type + dimensions.
	 * Does NOT start the decoder helper — call initHelper() from process()
	 * after the FrameFactory has been rebuilt by the Module framework.
	 */
	bool setMetadata(framemetadata_sp& metadata, frame_sp frame)
	{
		auto frameType = metadata->getFrameType();

		if (frameType == FrameMetadata::FrameType::H264_DATA)
		{
			auto type = H264Utils::getNALUType((char*)frame->data());
			if (type != H264Utils::H264_NAL_TYPE_IDR_SLICE && type != H264Utils::H264_NAL_TYPE_SEQ_PARAM)
			{
				return false;
			}

			auto h264Metadata = FrameMetadataFactory::downcast<H264Metadata>(metadata);
			bool spsParsed = false;
			try
			{
				sps_pps_properties p;
				H264ParserUtils::parse_sps(((const char*)frame->data()) + 5,
				                           frame->size() > 5 ? frame->size() - 5 : frame->size(), &p);
				if (p.width > 0 && p.height > 0)
				{
					mWidth = p.width;
					mHeight = p.height;
					spsParsed = true;
				}
			}
			catch (const std::exception& ex)
			{
				LOG_INFO << "SPS parsing failed: " << ex.what();
			}

			if (!spsParsed)
			{
				if (h264Metadata->getWidth() > 0 && h264Metadata->getHeight() > 0)
				{
					mWidth = h264Metadata->getWidth();
					mHeight = h264Metadata->getHeight();
				}
				else
				{
					mWidth = 1280;
					mHeight = 720;
					LOG_ERROR << "No valid dimensions from SPS or metadata, using default: " << mWidth << "x" << mHeight;
				}
			}

#ifdef ARM64
			mCodecPixFmt = V4L2_PIX_FMT_H264;
#endif
			mNeedsHelperInit = true;
			return true;
		}
		else if (frameType == FrameMetadata::FrameType::HEVC_DATA)
		{
			auto type = H265Utils::getNALUType((char*)frame->data());
			if (!H265Utils::isIDR(type) && type != H265Utils::H265_NAL_TYPE::VPS && type != H265Utils::H265_NAL_TYPE::SPS)
			{
				return false;
			}

			mWidth = 1920;
			mHeight = 1080;
#ifdef ARM64
			mCodecPixFmt = V4L2_PIX_FMT_H265;
#endif
			mNeedsHelperInit = true;
			return true;
		}
		else
		{
			LOG_ERROR << "VideoDecoder: unsupported frame type " << frameType;
			return false;
		}
	}

	/*
	 * Called from process() AFTER the FrameFactory has been rebuilt.
	 * Starts the V4L2 capture thread (or NvCodec helper) and stores the
	 * send/makeFrame callbacks.
	 */
	bool initHelper(std::function<void(frame_sp&)> sendCb, std::function<frame_sp()> makeFrame)
	{
		if (!mNeedsHelperInit)
			return true;
		mNeedsHelperInit = false;

#ifdef ARM64
		helper.reset(new h264DecoderV4L2Helper());
		return helper->init(sendCb, makeFrame, mCodecPixFmt);
#else
		helper.reset(new H264DecoderNvCodecHelper(mWidth, mHeight));
		return helper->init(sendCb, makeFrame);
#endif
	}

	void compute(void* inputFrameBuffer, size_t inputFrameSize, uint64_t inputFrameTS)
	{
		if (helper != nullptr)
		{
			helper->process(inputFrameBuffer, inputFrameSize, inputFrameTS);
		}
	}

#ifdef ARM64
	void closeAllThreads(frame_sp eosFrame)
	{
		if (helper != nullptr)
		{
			helper->closeAllThreads(eosFrame);
			helper.reset();
		}
	}
#endif

	int mWidth;
	int mHeight;

private:
#ifdef ARM64
	boost::shared_ptr<h264DecoderV4L2Helper> helper;
	uint32_t mCodecPixFmt;
#else
	boost::shared_ptr<H264DecoderNvCodecHelper> helper;
#endif
	bool mNeedsHelperInit;
	std::mutex mFramesMutex;
	std::deque<frame_sp> mDecodedFrames;
};

VideoDecoder::VideoDecoder(VideoDecoderProps _props)
	: Module(TRANSFORM, "VideoDecoder", _props), mShouldTriggerSOS(true), mHelperReady(false), mProps(_props)
{
	mDetail.reset(new Detail());
#ifdef ARM64
	mOutputMetadata = boost::shared_ptr<FrameMetadata>(new RawImageMetadata(FrameMetadata::MemType::DMABUF));
#else
	mOutputMetadata = boost::shared_ptr<FrameMetadata>(new RawImagePlanarMetadata(RawImageMetadata::MemType::HOST));
#endif
	mOutputPinId = Module::addOutputPin(mOutputMetadata);
}

VideoDecoder::~VideoDecoder() {}

bool VideoDecoder::init()
{
	if (!Module::init())
	{
		return false;
	}
	return true;
}

bool VideoDecoder::term()
{
#ifdef ARM64
	auto eosFrame = frame_sp(new EoSFrame());
	mDetail->closeAllThreads(eosFrame);
#endif
	mDetail.reset();
	return Module::term();
}

bool VideoDecoder::validateInputPins()
{
	if (getNumberOfInputPins() != 1)
	{
		LOG_ERROR << "VideoDecoder: supports only one input pin. Actual: " << getNumberOfInputPins();
		return false;
	}

	framemetadata_sp metadata = getFirstInputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::FrameType::H264_DATA && frameType != FrameMetadata::FrameType::HEVC_DATA)
	{
		LOG_ERROR << "VideoDecoder: input must be H264_DATA or HEVC_DATA. Actual: " << frameType;
		return false;
	}

	return true;
}

bool VideoDecoder::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{
		LOG_ERROR << "VideoDecoder: supports only one output pin. Actual: " << getNumberOfOutputPins();
		return false;
	}
	return true;
}

void VideoDecoder::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);
}

bool VideoDecoder::processEOS(string& pinId)
{
#ifdef ARM64
	auto eosFrame = frame_sp(new EoSFrame());
	mDetail->closeAllThreads(eosFrame);
#endif
	/* Drain any buffered decoded frames. */
	sendDecodedFrames();
	mShouldTriggerSOS = true;
	mHelperReady = false;
	return true;
}

bool VideoDecoder::shouldTriggerSOS()
{
	return mShouldTriggerSOS;
}

void VideoDecoder::flushQue()
{
	Module::flushQue();
}

bool VideoDecoder::handleCommand(Command::CommandType type, frame_sp& frame)
{
	return true;
}

bool VideoDecoder::processSOS(frame_sp& frame)
{
	auto metadata = frame->getMetadata();
	auto ret = mDetail->setMetadata(metadata, frame);

	if (ret)
	{
		mShouldTriggerSOS = false;
		mHelperReady = false; /* initHelper will be called on the next process() */
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

void VideoDecoder::sendDecodedFrames()
{
	auto decoded = mDetail->drainDecodedFrames();
	for (auto& outFrame : decoded)
	{
		frame_container frames;
		frames.insert(make_pair(mOutputPinId, outFrame));
		Module::send(frames);
	}
}

bool VideoDecoder::process(frame_container& frames)
{
	/*
	 * On the first process() call after processSOS(), the Module framework has
	 * already rebuilt the FrameFactory with the proper DMABUF metadata.
	 * Only NOW it is safe to start the V4L2 capture thread.
	 */
	if (!mHelperReady)
	{
		auto sendCb  = [this](frame_sp& f) { mDetail->bufferDecodedFrame(f); };
		auto makeFrm = [this]() -> frame_sp { return makeFrame(); };
		if (!mDetail->initHelper(sendCb, makeFrm))
		{
			LOG_ERROR << "VideoDecoder: initHelper failed";
			return false;
		}
		mHelperReady = true;
	}

	/* Drain decoded frames buffered by the capture thread. */
	sendDecodedFrames();

	/* Feed the current input frame to the decoder. */
	auto frame = frames.begin()->second;
	mDetail->compute(frame->data(), frame->size(), frame->timestamp);

	/* Drain again — capture thread may have produced output synchronously. */
	sendDecodedFrames();
	return true;
}
