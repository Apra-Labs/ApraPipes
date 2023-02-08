#include <cstdint>
extern "C"
{
#include <libavutil/motion_vector.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
}
#include "H264MotionExtractorXForm.h"
#include "H264FrameDemuxer.h"
#include "H264Utils.h"
#include "H264ParserUtils.h"
#include "Frame.h"
#include <fstream>

class MotionExtractor::Detail
{
public:
    Detail(MotionExtractorProps props) 
    {};
    ~Detail() 
    {
        avcodec_free_context(&video_dec_ctx);
    }

    int initDecoder()
    {
		int ret;
		AVCodecContext* dec_ctx = NULL;
		AVCodec* dec = NULL;
		AVDictionary* opts = NULL;
		dec = avcodec_find_decoder(AV_CODEC_ID_H264);

		dec_ctx = avcodec_alloc_context3(dec);
		if (!dec_ctx) 
        {
			LOG_ERROR << stderr << "Failed to allocate codec\n";
			return AVERROR(EINVAL);
		}
		/* Init the video decoder */
		av_dict_set(&opts, "flags2", "+export_mvs", 0);
		ret = avcodec_open2(dec_ctx, dec, &opts);
		av_dict_free(&opts);
		if (ret < 0) 
        {
            LOG_ERROR << stderr << "Failed to open %s codec\n";
			return ret;
		}

		video_dec_ctx = dec_ctx;
    }

    void decodeAndGetMV(frame_sp inFrame)
    {
		int ret = 0;
		AVPacket* pkt = NULL;

		frame = av_frame_alloc();
		if (!frame) {
			fprintf(stderr, "Could not allocate frame\n");
			ret = AVERROR(ENOMEM);
		}

		pkt = av_packet_alloc();
		if (!pkt) {
			fprintf(stderr, "Could not allocate AVPacket\n");
			ret = AVERROR(ENOMEM);
		}
		fpOut.open(outFileName, std::ios::out | std::ios::binary | std::ios::app);
		fpOut << "framenum , source , blockw , blockh , srcx , srcy , dstx , dsty , flags , motion_x  ,motion_y , motion_scale\n";
		fpOut.close();
		ret = decode_packet(pkt, inFrame);
		av_packet_unref(pkt);
    }

    int decode_packet(AVPacket* pkt,frame_sp inFrame)
    {       
        fpOut.open(outFileName, std::ios::out | std::ios::binary | std::ios::app);

        pkt->data = (uint8_t*)inFrame->data();
        pkt->size = (int)inFrame->size();

        int ret = avcodec_send_packet(video_dec_ctx, pkt);
        if (ret < 0) 
        {
            LOG_ERROR << stderr <<  "Error while sending a packet to the decoder: %s\n";
            return ret;
        }

        while (ret >= 0) 
        {
            ret = avcodec_receive_frame(video_dec_ctx, frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) 
            {
                break;
            }
            else if (ret < 0) 
            {
                LOG_ERROR << stderr <<  "Error while receiving a frame from the decoder: %s\n";
                return ret;
            }

            if (ret >= 0) 
            {
                int i;
                AVFrameSideData* sd;

                video_frame_count++;
                sd = av_frame_get_side_data(frame, AV_FRAME_DATA_MOTION_VECTORS);
                //sd = av_frame_get_side_data(frame, AV_FRAME_DATA_MOTION_VECTORS);
                if (sd) 
                {
                    const AVMotionVector* mvs = (const AVMotionVector*)sd->data;
                    for (i = 0; i < sd->size / sizeof(*mvs); i++) 
                    {
                        const AVMotionVector* mv = &mvs[i];
                        fpOut << video_frame_count << " , " << mv->source << " , " <<
                            static_cast<int>(mv->w) << ", " << static_cast<int>(mv->h) << " , " << mv->src_x << " , " << mv->src_y << " , "
                            << mv->dst_x << " , " << mv->dst_y << " , " << mv->flags << " , " <<
                            mv->motion_x << " , " << mv->motion_y << " , " << mv->motion_scale;
                        fpOut << std::endl;
                    }
                }
                av_frame_unref(frame);
            }
        }
        fpOut.close();
        return 0;
    }

private:
    AVFormatContext* fmt_ctx = NULL;
    AVCodecContext* video_dec_ctx = NULL;
    const char* outFileName = "C:/Users/developer/workspace/ffmpegGeneratedVect.txt";
    AVFrame* frame = NULL;
    int video_frame_count = 0;
    int count = 0;
    int fCount = 0;
    std::ofstream fpOut;
};


MotionExtractor::MotionExtractor(MotionExtractorProps props) : Module(TRANSFORM, "MotionExtractor", props)
{
    mDetail.reset(new Detail(props));
    auto outputMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::H264_DATA));
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
    return true;
}

bool MotionExtractor::validateOutputPins()
{
    if (getNumberOfOutputPins() != 1)
    {
        LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
        return false;
    }

   /* framemetadata_sp metadata = getFirstOutputMetadata();
    FrameMetadata::FrameType frameType = metadata->getFrameType();
    if (frameType != FrameMetadata::ENCODED_IMAGE)
    {
        LOG_ERROR << "<" << getId() << ">::validateOutputPins input frameType is expected to be ENCODED_IMAGE. Actual<" << frameType << ">";
        return false;
    }*/

    return true;
}

bool MotionExtractor::shouldTriggerSOS()
{
    return false;
}

bool MotionExtractor::process(frame_container& frames)
{

    auto frame = frames.begin()->second;
    
    mDetail->decodeAndGetMV(frame);

    frames.insert(make_pair(mOutputPinId, frame));
    send(frames);
    cc++;
    if (cc == 231)
    {
        auto eosFrame = frame_sp(new EoSFrame());
        mDetail->decodeAndGetMV(eosFrame);
    }
    return true;
}

bool MotionExtractor::processSOS(frame_sp& frame)
{
    return true;
}
//end:
//    avcodec_free_context(&video_dec_ctx);
//    avformat_close_input(&fmt_ctx);
//    av_frame_free(&frame);
//    av_packet_free(&pkt);
//    return ret < 0;
//}