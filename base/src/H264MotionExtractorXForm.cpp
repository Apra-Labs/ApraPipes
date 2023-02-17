#include <cstdint>
extern "C"
{
#include <libavutil/motion_vector.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil\imgutils.h>
#include <libswscale\swscale.h>
}
#include "H264MotionExtractorXForm.h"
#include "H264Metadata.h"
#include "fstream"

class MotionExtractor::Detail
{
public:
	Detail(MotionExtractorProps props, std::function<frame_sp(size_t)> _makeFrame, std::function<frame_sp(size_t size, string& pinId)> _makeFrameWithPinID)
	{
		makeFrameWithPinId = _makeFrameWithPinID;
		makeFrame = _makeFrame;
	};
	~Detail()
	{
		avcodec_free_context(&decoderContext);
	}

	void setProps(MotionExtractorProps props)
	{
		sendRgbFrame = props.sendRgbFrame;
	}

	void initDecoder()
	{
		int ret;
		AVCodec* dec = NULL;
		AVDictionary* opts = NULL;
		dec = avcodec_find_decoder(AV_CODEC_ID_H264);

		decoderContext = avcodec_alloc_context3(dec);
		if (!decoderContext)
		{
			throw AIPException(AIP_FATAL, "Failed to allocate codec");
		}
		/* Init the decoder */
		av_dict_set(&opts, "flags2", "+export_mvs", 0);
		ret = avcodec_open2(decoderContext, dec, &opts);
		av_dict_free(&opts);
		if (ret < 0)
		{
			throw AIPException(AIP_FATAL, "failed open decoder");
		}
	}

	void getMotionVectors(frame_sp inFrame, frame_sp& outFrame, frame_sp& rgbFrame)
	{
		int ret = 0;
		AVPacket* pkt = NULL;

		avFrame = av_frame_alloc();
		if (!avFrame)
		{
			LOG_ERROR << "Could not allocate frame\n";
		}

		pkt = av_packet_alloc();
		if (!pkt)
		{
			LOG_ERROR << "Could not allocate AVPacket\n";
		}
		ret = getMotionVectors(pkt, inFrame, outFrame, rgbFrame);
		av_packet_unref(pkt);
		av_frame_free(&avFrame);
	}

	int getMotionVectors(AVPacket* pkt, frame_sp inFrame, frame_sp& outFrame, frame_sp& rgbFrame)
	{
		pkt->data = (uint8_t*)inFrame->data();
		pkt->size = (int)inFrame->size();

		int ret = avcodec_send_packet(decoderContext, pkt);
		if (ret < 0)
		{
			LOG_ERROR << stderr << "Error while sending a packet to the decoder: %s\n";
			return ret;
		}

		while (ret >= 0)
		{
			ret = avcodec_receive_frame(decoderContext, avFrame);
			if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
			{
				break;
			}
			else if (ret < 0)
			{
				LOG_ERROR << stderr << "Error while receiving a frame from the decoder: %s\n";
				return ret;
			}

			if (sendRgbFrame)
			{
				SwsContext* sws_context = sws_getContext(
					decoderContext->width, decoderContext->height, decoderContext->pix_fmt,
					decoderContext->width, decoderContext->height, AV_PIX_FMT_BGR24,
					SWS_BICUBIC, NULL, NULL, NULL);
				if (!sws_context) {
					// Handle error
				}
				rgbFrame = makeFrameWithPinId(mWidth * mHeight * 3, rawFramePinId);

				int dstStrides[AV_NUM_DATA_POINTERS];
				dstStrides[0] = decoderContext->width * 3; // Assuming RGB format
				uint8_t* dstData[AV_NUM_DATA_POINTERS];
				dstData[0] = static_cast<uint8_t*>(rgbFrame->data());

				sws_scale(sws_context, avFrame->data, avFrame->linesize, 0, decoderContext->height, dstData, dstStrides);
			}
			else
			{
				rgbFrame = makeFrameWithPinId(0, rawFramePinId);
			}
			if (ret >= 0)
			{
				int i;
				AVFrameSideData* sideData;

				sideData = av_frame_get_side_data(avFrame, AV_FRAME_DATA_MOTION_VECTORS);

				if (sideData)
				{
					const AVMotionVector* motionVectors = (const AVMotionVector*)sideData->data;
					auto mvsSize = sizeof(*motionVectors);
					outFrame = makeFrame(sideData->size);
					memcpy(outFrame->data(), motionVectors, sideData->size);
				}
				else
				{
					outFrame = makeFrame(0);
				}
				av_frame_unref(avFrame);
			}
		}
		return 0;
	}
public:
	int mWidth = 0;
	int mHeight = 0;
	std::string rawFramePinId;
private:
	AVFormatContext* fmt_ctx = NULL;
	AVFrame* avFrame = NULL;
	AVCodecContext* decoderContext = NULL;
	std::function<frame_sp(size_t)> makeFrame;
	std::function<frame_sp(size_t size, string& pinId)> makeFrameWithPinId;
	bool sendRgbFrame = false;
};


MotionExtractor::MotionExtractor(MotionExtractorProps props) : Module(TRANSFORM, "MotionExtractor", props)
{
	mDetail.reset(new Detail(props, [&](size_t size) -> frame_sp {return makeFrame(size); }, [&](size_t size, string& pinId) -> frame_sp { return makeFrame(size, pinId); }));
	auto outputMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::MOTION_VECTOR_DATA));
	rawOutputMetadata = framemetadata_sp(new RawImageMetadata());
	motionVectorPinId = addOutputPin(outputMetadata);
	mDetail->rawFramePinId = addOutputPin(rawOutputMetadata);
}

bool MotionExtractor::init()
{
	mDetail->initDecoder();
	return Module::init();
}

bool MotionExtractor::term()
{
	return Module::term();
}

bool MotionExtractor::validateInputPins()
{
	if (getNumberOfInputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstInputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::H264_DATA)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be H264_DATA. Actual<" << frameType << ">";
		return false;
	}
	return true;
}

bool MotionExtractor::validateOutputPins()
{
	auto size = getNumberOfOutputPins();
	if (getNumberOfOutputPins() > 2)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstOutputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::MOTION_VECTOR_DATA)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input frameType is expected to be MOTION_VECTOR_DATA. Actual<" << frameType << ">";
		return false;
	}

	return true;
}

bool MotionExtractor::shouldTriggerSOS()
{
	return true;
}

bool MotionExtractor::process(frame_container& frames)
{

	auto inFrame = frames.begin()->second;
	frame_sp motionVectorFrame;
	frame_sp rgbFrame;
	mDetail->getMotionVectors(inFrame, motionVectorFrame, rgbFrame);

	frames.insert(make_pair(motionVectorPinId, motionVectorFrame));
	frames.insert(make_pair(mDetail->rawFramePinId, rgbFrame));
	send(frames);
	return true;
}

void MotionExtractor::setMetadata(framemetadata_sp& metadata)
{
	if (!metadata->isSet())
	{
		return;
	}

	auto rawMetadata = FrameMetadataFactory::downcast<H264Metadata>(metadata);
	mDetail->mWidth = rawMetadata->getWidth();
	mDetail->mHeight = rawMetadata->getHeight();
	RawImageMetadata outputMetadata(rawMetadata->getWidth(), rawMetadata->getHeight(), ImageMetadata::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true);
	auto rawOutMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(rawOutputMetadata);
	rawOutMetadata->setData(outputMetadata);
}
bool MotionExtractor::processSOS(frame_sp& frame)
{
	auto metadata = frame->getMetadata();
	setMetadata(metadata);

	return true;
}

bool MotionExtractor::handlePropsChange(frame_sp& frame)
{
	MotionExtractorProps props;
	auto ret = Module::handlePropsChange(frame, props);
	mDetail->setProps(props);
	return ret;
}

void MotionExtractor::setProps(MotionExtractorProps& props)
{
	Module::addPropsToQueue(props);
}