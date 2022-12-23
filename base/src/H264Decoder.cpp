#include "H264Decoder.h"
#ifdef _WIN64
#include "H264DecoderNvCodecHelper.h"
#endif

#ifdef ARM64
#include "H264DecoderV4L2Helper.h"
#endif
#include "H264ParserUtils.h"
#include "FrameMetadata.h"
#include "H264Metadata.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"

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
		if (metadata->getFrameType() == FrameMetadata::FrameType::H264_DATA)
		{
			sps_pps_properties p;
			H264ParserUtils::parse_sps(((const char*)frame->data()) + 5, frame->size() > 5 ? frame->size() - 5 : frame->size(), &p);
			mWidth = p.width;
			mHeight = p.height;

			auto h264Metadata = framemetadata_sp(new H264Metadata(mWidth, mHeight));
			auto rawOutMetadata = FrameMetadataFactory::downcast<H264Metadata>(h264Metadata);
			rawOutMetadata->setData(*rawOutMetadata);
		}

		else
		{
			throw AIPException(AIP_NOTIMPLEMENTED, "Unknown frame type");
		}

#ifdef _WIN64
		helper.reset(new H264DecoderNvCodecHelper(mWidth, mHeight));
		return helper->init(send, makeFrame);

#elif ARM64
		helper.reset(new h264DecoderV4L2Helper());
		return helper->init(send, makeFrame);
#endif
	}

	bool compute(frame_sp& frame)
	{
		return helper->process(frame);
	}
public:
	int mWidth;
	int mHeight;
private:

#ifdef _WIN64
	boost::shared_ptr<H264DecoderNvCodecHelper> helper;
#elif ARM64
	boost::shared_ptr<h264DecoderV4L2Helper> helper;
#endif
};

H264Decoder::H264Decoder(H264DecoderProps _props) : Module(TRANSFORM, "H264Decoder", _props), mShouldTriggerSOS(true), mProps(_props)
{
	mDetail.reset(new Detail(mProps));
}

H264Decoder::~H264Decoder() {}

bool H264Decoder::validateInputPins()
{
	auto numberOfInputPins = getNumberOfInputPins();
	if (numberOfInputPins > 2)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
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
#ifdef _WIN64
	mOutputMetadata = boost::shared_ptr<FrameMetadata>(new RawImagePlanarMetadata(RawImageMetadata::MemType::HOST));

#elif ARM64
	mOutputMetadata = boost::shared_ptr<FrameMetadata>(new RawImagePlanarMetadata(FrameMetadata::MemType::DMABUF));
#endif
	mOutputPinId = Module::addOutputPin(mOutputMetadata);
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
	mDetail.reset();

	return Module::term();
}

bool H264Decoder::process(frame_container& frames)
{
	auto frame = frames.cbegin()->second;
	mDetail->compute(frame);
	return true;
}

bool H264Decoder::processSOS(frame_sp& frame)
{
	auto metadata = frame->getMetadata();
	mDetail->setMetadata(metadata, frame,
		[&](frame_sp& outputFrame) {
			frame_container frames;
			frames.insert(make_pair(mOutputPinId, outputFrame));
			send(frames);
		}, [&]() -> frame_sp {return makeFrame(); }
		);
	mShouldTriggerSOS = false;
	auto rawOutMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(mOutputMetadata);
#ifdef _WIN64
	RawImagePlanarMetadata OutputMetadata(mDetail->mWidth, mDetail->mHeight, ImageMetadata::YUV420, size_t(0), CV_8U, FrameMetadata::HOST);
	rawOutMetadata->setData(OutputMetadata);
	return true;
#elif ARM64
	RawImagePlanarMetadata OutputMetadata(mDetail->mWidth, mDetail->mHeight, ImageMetadata::ImageType::NV12, 128, CV_8U, FrameMetadata::MemType::DMABUF);
	rawOutMetadata->setData(OutputMetadata);
	return true;
#endif
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
