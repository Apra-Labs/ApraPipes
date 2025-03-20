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

	bool setMetadata(framemetadata_sp& metadata, frame_sp frame, std::function<void(frame_sp&)> send, std::function<frame_sp()> makeFrame, std::function<void()> flushModuleQueue)
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
		
#ifdef ARM64
		helper.reset(new h264DecoderV4L2Helper());
		return helper->init(send, makeFrame, flushModuleQueue);
#else
		helper.reset(new H264DecoderNvCodecHelper(mWidth, mHeight));
		return helper->init(send, makeFrame);//
#endif
	}

	void compute(frame_sp& frame)
	{
		if(frame->size())
		{
			helper->process(frame);
		}
	}
	
	void intitliazeSpeedElement(int _playbackFps, float _playbackSpeed, int _gop)
	{
		LOG_ERROR << "Decoder is initializing speed factor.";
		helper->intitliazeSpeed(_playbackFps, _playbackSpeed, _gop);
	}

	void terminateHelper()
	{
		LOG_ERROR << "Should Free all the resources ";
		helper->term_helper();
	}
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
	LOG_ERROR << "calling decoder term";
#ifdef ARM64
	mDetail->terminateHelper();
#endif
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
	// change bool to true
	LOG_ERROR << "================Got SOS FRAME ==============";
	mDetail->setMetadata(metadata, frame, [&](frame_sp &outputFrame)
						 {
			frame_container frames;
			frames.insert(make_pair(mOutputPinId, outputFrame));
			send(frames); }, [&]() -> frame_sp
						 { return makeFrame(); }, [&]()
						 { flushQue(); });
	mShouldTriggerSOS = false;
	auto rawOutMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(mOutputMetadata);

#ifdef ARM64
	RawImagePlanarMetadata OutputMetadata(mDetail->mWidth, mDetail->mHeight, ImageMetadata::ImageType::NV12, 128, CV_8U, FrameMetadata::MemType::DMABUF);
#else
	RawImagePlanarMetadata OutputMetadata(mDetail->mWidth, mDetail->mHeight, ImageMetadata::YUV420, size_t(0), CV_8U, FrameMetadata::HOST);
#endif

	rawOutMetadata->setData(OutputMetadata);
	return true;
}

bool H264Decoder::shouldTriggerSOS()
{
	return mShouldTriggerSOS;
}

bool H264Decoder::processEOS(string& pinId)
{
	LOG_ERROR << "Got EOS Frame";
	auto frame = frame_sp(new EmptyFrame());
	mDetail->compute(frame);
	mDetail->terminateHelper();
	mShouldTriggerSOS = true;
	return true;
}

bool H264Decoder::handleCommand(Command::CommandType type, frame_sp& frame)
{
	if (type == Command::CommandType::DecoderPlaybackSpeed)
	{
		DecoderPlaybackSpeed cmd;
		getCommand(cmd, frame);
		// currentFps = cmd.playbackFps;
		// playbackSpeed = cmd.playbackSpeed;
		// gop = cmd.gop;
		
		mDetail->intitliazeSpeedElement(cmd.playbackFps, cmd.playbackSpeed, cmd.gop);
	}
	else if (type == Command::CommandType::DecoderEOS)
	{ 
		DecoderEOS decEos;
		getCommand(decEos, frame);
		processEOS(mOutputPinId);
	}
	else
	{
		return Module::handleCommand(type, frame);
	}
}

bool H264Decoder::decoderEos()
{
	DecoderEOS cmd;
	return queueCommand(cmd);
}

void H264Decoder::changeDecoderSpeed(int _playbackFps, float _playbackSpeed, int _gop)
{
	DecoderPlaybackSpeed cmd;
	cmd.playbackFps = _playbackFps;
	cmd.playbackSpeed = _playbackSpeed;
	cmd.gop = _gop;

	Module::queueCommand(cmd, true);
}

void H264Decoder::flushQue()
{
	Module::flushQue();
}
 