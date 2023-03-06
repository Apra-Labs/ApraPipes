#include <cstdint>
#include <boost/foreach.hpp>
extern "C"
{
#include <libavutil/motion_vector.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}
#include "MotionVectorExtractor.h"
#include "H264Metadata.h"
#include "H264ParserUtils.h"

class MotionVectorExtractor::Detail
{
public:
	Detail(MotionVectorExtractorProps props, std::function<frame_sp(size_t)> _makeFrame, std::function<frame_sp(size_t size, string& pinId)> _makeFrameWithPinId)
	{
		makeFrameWithPinId = _makeFrameWithPinId;
		makeFrame = _makeFrame;
		sendDecodedFrame = props.sendDecodedFrame;
	};
	~Detail()
	{
		avcodec_free_context(&decoderContext);
	}

	void setProps(MotionVectorExtractorProps props)
	{
		sendDecodedFrame = props.sendDecodedFrame;
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

	void getMotionVectors(frame_container& frames, frame_sp& outFrame, frame_sp& decodedFrame)
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
		ret = decodeAndGetMotionVectors(pkt, frames, outFrame, decodedFrame);
		av_packet_free(&pkt);
		av_frame_free(&avFrame);
	}

	int decodeAndGetMotionVectors(AVPacket* pkt, frame_container& frames, frame_sp& outFrame, frame_sp& decodedFrame)
	{
		auto inFrame = frames.begin()->second;
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
				outFrame = makeFrame(0);
				break;
			}
			else if (ret < 0)
			{
				LOG_ERROR << stderr << "Error while receiving a frame from the decoder: %s\n";
				return ret;
			}

			if (sendDecodedFrame)
			{
				SwsContext* sws_context = sws_getContext(
					decoderContext->width, decoderContext->height, decoderContext->pix_fmt,
					decoderContext->width, decoderContext->height, AV_PIX_FMT_BGR24,
					SWS_BICUBIC | SWS_FULL_CHR_H_INT, NULL, NULL, NULL);
				if (!sws_context) {
					// Handle error
				}
				decodedFrame = makeFrameWithPinId(mWidth * mHeight * 3, rawFramePinId);

				int dstStrides[AV_NUM_DATA_POINTERS];
				dstStrides[0] = decoderContext->width * 3; // Assuming BGR format
				uint8_t* dstData[AV_NUM_DATA_POINTERS];
				dstData[0] = static_cast<uint8_t*>(decodedFrame->data());

				sws_scale(sws_context, avFrame->data, avFrame->linesize, 0, decoderContext->height, dstData, dstStrides);

				frames.insert(make_pair(rawFramePinId, decodedFrame));
			}
			if (ret >= 0)
			{
				AVFrameSideData* sideData;

				sideData = av_frame_get_side_data(avFrame, AV_FRAME_DATA_MOTION_VECTORS);

				if (sideData)
				{
					const AVMotionVector* motionVectors = (const AVMotionVector*)sideData->data;
					outFrame = makeFrame(sideData->size);
					memcpy(outFrame->data(), motionVectors, sideData->size);
				}
				else
				{
					outFrame = makeFrame(0);
				}
				av_packet_unref(pkt);
				av_frame_unref(avFrame);
				return 0;
			}
		}
		return 0;
	}
public:
	int mWidth = 0;
	int mHeight = 0;
	std::string rawFramePinId;
private:
	AVFrame* avFrame = NULL;
	AVCodecContext* decoderContext = NULL;
	std::function<frame_sp(size_t)> makeFrame;
	std::function<frame_sp(size_t size, string& pinId)> makeFrameWithPinId;
	bool sendDecodedFrame = false;
};


MotionVectorExtractor::MotionVectorExtractor(MotionVectorExtractorProps props) : Module(TRANSFORM, "MotionVectorExtractor", props)
{
	mDetail.reset(new Detail(props, [&](size_t size) -> frame_sp {return makeFrame(size); }, [&](size_t size, string& pinId) -> frame_sp { return makeFrame(size, pinId); }));
	auto motionVectorOutputMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::MOTION_VECTOR_DATA));
	rawOutputMetadata = framemetadata_sp(new RawImageMetadata());
	motionVectorPinId = addOutputPin(motionVectorOutputMetadata);
	mDetail->rawFramePinId = addOutputPin(rawOutputMetadata);
}

bool MotionVectorExtractor::init()
{
	mDetail->initDecoder();
	return Module::init();
}

bool MotionVectorExtractor::term()
{
	return Module::term();
}

bool MotionVectorExtractor::validateInputPins()
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

bool MotionVectorExtractor::validateOutputPins()
{
	auto size = getNumberOfOutputPins();
	if (getNumberOfOutputPins() > 2)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 2. Actual<" << getNumberOfOutputPins() << ">";
		return false;
	}

	pair<string, framefactory_sp> me; // map element	
	auto framefactoryByPin = getOutputFrameFactory();
	BOOST_FOREACH(me, framefactoryByPin)
	{
		FrameMetadata::FrameType frameType = me.second->getFrameMetadata()->getFrameType();
		if (frameType != FrameMetadata::MOTION_VECTOR_DATA && frameType != FrameMetadata::RAW_IMAGE)
		{
			LOG_ERROR << "<" << getId() << ">::validateOutputPins input frameType is expected to be MOTION_VECTOR_DATA or RAW_IMAGE. Actual<" << frameType << ">";
			return false;
		}
	}

	return true;
}

bool MotionVectorExtractor::shouldTriggerSOS()
{
	return mShouldTriggerSOS;
}

bool MotionVectorExtractor::process(frame_container& frames)
{
	frame_sp motionVectorFrame;
	frame_sp decodedFrame;
	mDetail->getMotionVectors(frames, motionVectorFrame, decodedFrame);

	frames.insert(make_pair(motionVectorPinId, motionVectorFrame));
	send(frames);
	return true;
}

void MotionVectorExtractor::setMetadata(frame_sp frame)
{
	auto metadata = frame->getMetadata();
	if (!metadata->isSet())
	{
		return;
	}
	sps_pps_properties p;
	H264ParserUtils::parse_sps(((const char*)frame->data()) + 5, frame->size() > 5 ? frame->size() - 5 : frame->size(), &p);
	mDetail->mWidth = p.width;
	mDetail->mHeight = p.height;

	RawImageMetadata outputMetadata(mDetail->mWidth, mDetail->mHeight, ImageMetadata::BGR, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true);
	auto rawOutMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(rawOutputMetadata);
	rawOutMetadata->setData(outputMetadata);
}
bool MotionVectorExtractor::processSOS(frame_sp& frame)
{
	setMetadata(frame);
	mShouldTriggerSOS = false;
	return true;
}

bool MotionVectorExtractor::handlePropsChange(frame_sp& frame)
{
	MotionVectorExtractorProps props;
	auto ret = Module::handlePropsChange(frame, props);
	mDetail->setProps(props);
	return ret;
}

void MotionVectorExtractor::setProps(MotionVectorExtractorProps& props)
{
	Module::addPropsToQueue(props);
}