#include <cstdint>
extern "C"
{
#include <libavutil/motion_vector.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}
#include "H264MotionExtractorXForm.h"

class MotionExtractor::Detail
{
public:
	Detail(MotionExtractorProps props, std::function<frame_sp(size_t)> _makeFrame)
	{
		makeFrame = _makeFrame;
	};
	~Detail()
	{
		avcodec_free_context(&decoderContext);
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

	void decodeAndGetMV(frame_sp inFrame, frame_sp& outFrame)
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
		ret = decodePacket(pkt, inFrame, outFrame);
		av_packet_unref(pkt);
		av_frame_free(&avFrame);
	}

	int decodePacket(AVPacket* pkt, frame_sp inFrame, frame_sp& outFrame)
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

			if (ret >= 0)
			{
				int i;
				AVFrameSideData* sideData;

				sideData = av_frame_get_side_data(avFrame, AV_FRAME_DATA_MOTION_VECTORS);

				if (sideData)
				{
					const AVMotionVector* motionVectors = (const AVMotionVector*)sideData->data;
					auto mvsSize = sizeof(*motionVectors);
					outFrame = makeFrame(sideData->size / mvsSize);
					memcpy(outFrame->data(), motionVectors, sideData->size / mvsSize);
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

private:
	AVFormatContext* fmt_ctx = NULL;
	AVFrame* avFrame = NULL;
	AVCodecContext* decoderContext = NULL;
	std::function<frame_sp(size_t)> makeFrame;
};


MotionExtractor::MotionExtractor(MotionExtractorProps props) : Module(TRANSFORM, "MotionExtractor", props)
{
	mDetail.reset(new Detail(props, [&](size_t size) -> frame_sp {return makeFrame(size); }));
	auto outputMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::MOTION_VECTOR_DATA));
	mOutputPinId = addOutputPin(outputMetadata);
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
	if (getNumberOfOutputPins() != 1)
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
	return false;
}

bool MotionExtractor::process(frame_container& frames)
{

	auto inFrame = frames.begin()->second;
	frame_sp outFrame;
	mDetail->decodeAndGetMV(inFrame, outFrame);

	frames.insert(make_pair(mOutputPinId, outFrame));
	send(frames);
	return true;
}