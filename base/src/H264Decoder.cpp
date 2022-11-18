#include "H264Decoder.h"
#include "H264DecoderHelper.h"
#include "H264ParserUtils.h"
#include "FrameMetadata.h"
#include "H264Metadata.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"

class H264DecoderNvCodec::Detail
{
public:
	Detail(H264DecoderNvCodecProps& _props)
	{
	}

	~Detail()
	{
		helper.reset();
	}

	bool setMetadata(framemetadata_sp& metadata, frame_sp frame,std::function<void(frame_sp&)> send)
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

		helper.reset(new H264DecoderNvCodecHelper(mWidth, mHeight));
		return helper->init(send);
	}

	bool compute(frame_sp& frame, frame_sp outFrame)
	{
		return helper->process(frame,outFrame);
	}
public:
	int mWidth = 0;
	int mHeight = 0;
private:
	boost::shared_ptr<H264DecoderNvCodecHelper> helper;
};

H264DecoderNvCodec::H264DecoderNvCodec(H264DecoderNvCodecProps _props) : Module(TRANSFORM, "H264DecoderNvCodec", _props), mShouldTriggerSOS(true), props(_props)
{
	mDetail.reset(new Detail(props));
}

H264DecoderNvCodec::~H264DecoderNvCodec() {}

bool H264DecoderNvCodec::validateInputPins()
{
	auto numberOfInputPins = getNumberOfInputPins();
	if (numberOfInputPins > 2)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstInputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType!= FrameMetadata::FrameType::H264_DATA)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be H264_DATA. Actual<" << frameType << ">";
		return false;
	}

	FrameMetadata::MemType memType = metadata->getMemType();
	if (memType != FrameMetadata::MemType::HOST)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input memType is expected to be HOST_DEVICE. Actual<" << memType << ">";
		return false;
	}

	return true;
}

bool H264DecoderNvCodec::validateOutputPins()
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

void H264DecoderNvCodec::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);
	mOutputMetadata = boost::shared_ptr<FrameMetadata>(new RawImagePlanarMetadata(RawImageMetadata::MemType::HOST));
	
	mOutputPinId = Module::addOutputPin(mOutputMetadata);
}

bool H264DecoderNvCodec::init()
{
	if (!Module::init())
	{
		return false;
	}

	return true;
}

bool H264DecoderNvCodec::term()
{
	mDetail.reset();

	return Module::term();
}

bool H264DecoderNvCodec::process(frame_container& frames)
{
	auto frame = frames.cbegin()->second;
	auto outputFrame = makeFrame();
	mDetail->compute(frame,outputFrame);
	return true;
}

bool H264DecoderNvCodec::processSOS(frame_sp& frame)
{
	auto metadata = frame->getMetadata();
	mDetail->setMetadata(metadata, frame,
		[&](frame_sp& outputFrame) {
			frame_container frames;
			frames.insert(make_pair(mOutputPinId, outputFrame));
			send(frames);
		}
	);
	mShouldTriggerSOS = false;
	RawImagePlanarMetadata OutputMetadata(mDetail->mWidth, mDetail->mHeight, ImageMetadata::YUV420, size_t(0), CV_8U, FrameMetadata::HOST);
	auto rawOutMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(mOutputMetadata);
	rawOutMetadata->setData(OutputMetadata);
	return true;
}

bool H264DecoderNvCodec::shouldTriggerSOS()
{
	return mShouldTriggerSOS;
}

bool H264DecoderNvCodec::processEOS(string& pinId)
{
	auto frame = frame_sp(new EmptyFrame());
	auto outputFrame = makeFrame();
	mDetail->compute(frame, outputFrame);
	mShouldTriggerSOS = true;
	return true;
}