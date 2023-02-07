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
    ~Detail() { }

    void openVideo()
    {
        if (avformat_open_input(&fmt_ctx, src_filename, NULL, NULL) < 0) {
            fprintf(stderr, "Could not open source file %s\n", src_filename);
            exit(1);
        }
    }

    void openStreamInfo()
    {
        if (avformat_find_stream_info(fmt_ctx, NULL) < 0) {
            fprintf(stderr, "Could not find stream information\n");
            exit(1);
        }
    }

    int open_codec_context()
    {
        AVMediaType type = AVMEDIA_TYPE_VIDEO;
        int ret;
        AVStream* st;
        AVCodecContext* dec_ctx = NULL;
        AVCodec* dec = NULL;
        AVDictionary* opts = NULL;

        ret = av_find_best_stream(fmt_ctx, type, -1, -1, &dec, 0);
        if (ret < 0) {
            fprintf(stderr, "Could not find %s stream in input file '%s'\n",
                av_get_media_type_string(type), src_filename);
            return ret;
        }
        else {
            int stream_idx = ret;
            st = fmt_ctx->streams[stream_idx];

            dec_ctx = avcodec_alloc_context3(dec);
            if (!dec_ctx) {
                fprintf(stderr, "Failed to allocate codec\n");
                return AVERROR(EINVAL);
            }

            ret = avcodec_parameters_to_context(dec_ctx, st->codecpar);
            if (ret < 0) {
                fprintf(stderr, "Failed to copy codec parameters to codec context\n");
                return ret;
            }

            /* Init the video decoder */
            av_dict_set(&opts, "flags2", "+export_mvs", 0);
            ret = avcodec_open2(dec_ctx, dec, &opts);
            av_dict_free(&opts);
            if (ret < 0) {
                fprintf(stderr, "Failed to open %s codec\n",
                    av_get_media_type_string(type));
                return ret;
            }

            video_stream_idx = stream_idx;
            video_stream = fmt_ctx->streams[video_stream_idx];
            video_dec_ctx = dec_ctx;
        }

        return 0;
    }

    void dumpFormat()
    {
        av_dump_format(fmt_ctx, 0, src_filename, 0);
    }

    void decodeAndGetMV()
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
        fpOut << "framenum , source , blockw , blockh , srcx , srcy , dstx , dsty , flags , motion_x  ,motion_y , motion_scale\n" ;
        fpOut.close();
        /* read frames from the file */
        while (av_read_frame(fmt_ctx, pkt) >= 0) {
            if (pkt->stream_index == video_stream_idx)
                ret = decode_packet(pkt);
            av_packet_unref(pkt);
            if (ret < 0)
                break;
        }
    }

    int decode_packet(const AVPacket* pkt)
    {
       
        fpOut.open(outFileName, std::ios::out | std::ios::binary | std::ios::app);

        int ret = avcodec_send_packet(video_dec_ctx, pkt);
        if (ret < 0) {
            //fprintf(stderr, "Error while sending a packet to the decoder: %s\n", av_err2str(ret));
            return ret;
        }

        while (ret >= 0) {
            ret = avcodec_receive_frame(video_dec_ctx, frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                break;
            }
            else if (ret < 0) {
                //fprintf(stderr, "Error while receiving a frame from the decoder: %s\n", av_err2str(ret));
                return ret;
            }

            if (ret >= 0) {
                int i;
                AVFrameSideData* sd;

                video_frame_count++;
                sd = av_frame_get_side_data(frame, AV_FRAME_DATA_MOTION_VECTORS);
                if (sd) {
                    const AVMotionVector* mvs = (const AVMotionVector*)sd->data;
                    for (i = 0; i < sd->size / sizeof(*mvs); i++) {
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
    AVStream* video_stream = NULL;
    const char* src_filename = "C:/Users/developer/workspace/ApraPipesZaki/ApraPipes/data/Mp4_videos/h264_video/20221010/0012/1668063524439.mp4";
    const char* outFileName = "C:/Users/developer/workspace/ffmpegGeneratedVect.txt";
    int video_stream_idx = -1;
    AVFrame* frame = NULL;
    int video_frame_count = 0;
    int count = 0;
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
    mDetail->openVideo();
    mDetail->openStreamInfo();
    mDetail->open_codec_context();
    mDetail->dumpFormat();
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

//#define av_err2str	(	 	errnum	)	   av_make_error_string((char[AV_ERROR_MAX_STRING_SIZE]){0}, AV_ERROR_MAX_STRING_SIZE, errnum)


bool MotionExtractor::process(frame_container& frames)
{
    auto frame = frames.begin()->second;
    mDetail->decodeAndGetMV();

    frames.insert(make_pair(mOutputPinId, frame));
    send(frames);
    return false;

    /* flush cached frames */
    //decode_packet(NULL);
}
//end:
//    avcodec_free_context(&video_dec_ctx);
//    avformat_close_input(&fmt_ctx);
//    av_frame_free(&frame);
//    av_packet_free(&pkt);
//    return ret < 0;
//}