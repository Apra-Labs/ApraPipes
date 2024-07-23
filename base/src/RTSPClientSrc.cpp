#include "RTSPClientSrc.h"
#include "FrameMetadataFactory.h"
#include "FrameMetadata.h"
#include "H264Metadata.h"

using namespace std;


#include <iostream>
#include <string>

#include <thread>
#include <mutex>
#include <chrono>
#include "H264Utils.h"

extern "C"
{
#include <libavutil/mathematics.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
#include <libavutil/pixdesc.h>
#include <libavdevice/avdevice.h>
}

FrameMetadata::FrameType getFrameType_fromFFMPEG(AVMediaType avMediaType, AVCodecID avCodecID)
{
    if (avMediaType == AVMEDIA_TYPE_VIDEO)
    {
        switch (avCodecID)
        {
        case AV_CODEC_ID_H264:
            return FrameMetadata::H264_DATA;
        case AV_CODEC_ID_HEVC:
            return FrameMetadata::HEVC_DATA;
        case AV_CODEC_ID_MJPEG:
            return FrameMetadata::ENCODED_IMAGE;
        case AV_CODEC_ID_BMP:
            return FrameMetadata::BMP_IMAGE;
        default:
            return  FrameMetadata::GENERAL;
        }
    }
    return  FrameMetadata::GENERAL; //for everything else we may not have a match
}


class RTSPClientSrc::Detail
{
public:
    Detail(RTSPClientSrc* m,std::string path, bool useTCP) :myModule(m), path(path), bConnected(false), bUseTCP(useTCP){}
    ~Detail() { destroy(); }
    void destroy()
    {
        if(nullptr!= pFormatCtx)
            avformat_close_input(&pFormatCtx);
    }
    
    bool connect()
    {
        avformat_network_init();
        av_register_all();

        pFormatCtx = avformat_alloc_context();

        AVDictionary* avdic = NULL;

        av_dict_set(&avdic, "rtsp_transport", (bUseTCP)?"tcp":"udp", 0);
        av_dict_set(&avdic, "max_delay", "100", 0);

        if (avformat_open_input(&pFormatCtx, path.c_str(), NULL, &avdic) != 0)
        {
            LOG_ERROR << "can't open the URL." << path <<std::endl;
            bConnected = false;
            return bConnected;
        }

        if (avformat_find_stream_info(pFormatCtx, NULL) < 0)
        {
            LOG_ERROR << "can't find stream infomation" << std::endl;
            bConnected = false;
            return bConnected;
        }

        //this loop should check that each output pin is satisfied
        for (unsigned int i = 0; i < pFormatCtx->nb_streams; i++)
        {
            auto pCodecCtx = pFormatCtx->streams[i]->codec;
            LOG_INFO << av_get_media_type_string (pCodecCtx->codec_type)<<" Codec: " << avcodec_get_name(pCodecCtx->codec_id);
            auto fType=getFrameType_fromFFMPEG(pCodecCtx->codec_type,pCodecCtx->codec_id);
            string outPin=myModule->getOutputPinIdByType(fType);
            if (!outPin.empty())
            {
                streamsMap[i] = outPin;
                if (pCodecCtx->codec_type == AVMEDIA_TYPE_VIDEO)
                {
                    videoStream = i;
                    auto meta= FrameMetadataFactory::downcast<H264Metadata>(myModule->getOutputMetadata(outPin));
                    H264Metadata tmp(pCodecCtx->width, pCodecCtx->height, pCodecCtx->gop_size, pCodecCtx->max_b_frames);
                    meta->setData(tmp);
                }
            }
        }

        //by this time all the output pins should be satisfied if not we have a problem
        if (streamsMap.size() < myModule->getNumberOfOutputPins())
        {
            LOG_ERROR << "some output pins can not be satsified" << std::endl;
        }
        av_init_packet(&packet);
        bConnected = true;
        return bConnected;
    }

    frame_sp prependSpsPpsToFrame(std::string id)
    {
        auto spsPpsData = pFormatCtx->streams[0]->codec->extradata;
        auto spsPpsSize = pFormatCtx->streams[0]->codec->extradata_size;
        size_t totalFrameSize = packet.size + spsPpsSize;

        auto frm = myModule->makeFrame(totalFrameSize, id);
        uint8_t* frameData = static_cast<uint8_t*>(frm->data());
        memcpy(frameData, spsPpsData, spsPpsSize);
        frameData += spsPpsSize;
        memcpy(frameData, packet.data, packet.size);
        return frm;
    }

    bool readBuffer()
    {
        if(!initDone)
        {
            std::chrono::time_point<std::chrono::system_clock> t = std::chrono::system_clock::now();
            beginTs = std::chrono::duration_cast<std::chrono::milliseconds>(t.time_since_epoch());
            initDone = true;
        }
        frame_container outFrames;
        bool got_something = false;
        while(!got_something)
        {
            if (av_read_frame(pFormatCtx, &packet) >= 0)
            {
                if (videoStream >= 0) //source has video
                {
                    if (packet.stream_index == videoStream) //got video
                    {
                        got_something = true;
                    }
                }
                else
                {
                    got_something = true; //does not need video and got something
                }
                auto it = streamsMap.find(packet.stream_index);
                if (it != streamsMap.end()) { // so we have an interest in sending this
                    frame_sp frm;
                    auto naluType = H264Utils::getNALUType((const char*)packet.data);
                    if (naluType == H264Utils::H264_NAL_TYPE_SEI)
                    {
                        size_t offset = 0;
                        packet.data += 4;
                        packet.size -= 4;
                        H264Utils::getNALUnit((const char*)packet.data, packet.size, offset);
                        packet.data += offset - 4;
                        packet.size -= offset - 4;
                        frm = prependSpsPpsToFrame(it->second);
                    }
                    else if (naluType == H264Utils::H264_NAL_TYPE_IDR_SLICE)
                    {
                        frm = prependSpsPpsToFrame(it->second);
                    }
                    else
                    {
                        frm = myModule->makeFrame(packet.size, it->second);
                        memcpy(frm->data(), packet.data, packet.size);
                    }

                    std::chrono::time_point<std::chrono::system_clock> t = std::chrono::system_clock::now();
                	auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(t.time_since_epoch());
                    frm->timestamp = dur.count();
                    if (!outFrames.insert(make_pair(it->second, frm)).second)
                    {
                        LOG_WARNING << "oops! there is already another packet for pin " << it->second;
                    }
                    auto diff = dur - beginTs;
                    if(diff.count() > 1000)
                    {
                        currentCameraFps = frameCount;
                        frameCount = 0;
                        beginTs = dur;
                    }
                    frameCount++;
                }
                av_packet_unref(&packet);
            }
        }

        if(outFrames.size()>0)
           myModule->send(outFrames);
        return true;
    }

    bool isConncected() const { return bConnected; }
    int frameCount = 0;
    int currentCameraFps = 0;
private:
    AVPacket packet;
    AVFormatContext* pFormatCtx = nullptr;
    std::string path;
    bool bConnected;
    int videoStream=-1;
    bool bUseTCP;
    std::map<unsigned int, std::string> streamsMap;
    RTSPClientSrc* myModule;
	std::chrono::milliseconds beginTs;
	bool initDone = false;
};

RTSPClientSrc::RTSPClientSrc(RTSPClientSrcProps _props) : Module(SOURCE, "RTSPClientSrc", _props), mProps(_props)
{
    mDetail.reset(new Detail(this,mProps.rtspURL, mProps.useTCP));
}
RTSPClientSrc::~RTSPClientSrc() {
}
bool RTSPClientSrc::init() {
    if (mDetail->connect())
    {
        return Module::init();
    }
    return false;
}
bool RTSPClientSrc::term() {
    mDetail.reset();
    return true;
}
void RTSPClientSrc::setProps(RTSPClientSrcProps& props)
{
    mProps = props;
    //TBD need to also reset the whole connection
}
RTSPClientSrcProps RTSPClientSrc::getProps() {
    return mProps;
}

bool RTSPClientSrc::produce() { 
    return mDetail->readBuffer();
}
bool RTSPClientSrc::validateOutputPins() { 
    //smallest check at least one output pin should be there
    return this->getNumberOfOutputPins() > 0;
}
void RTSPClientSrc::notifyPlay(bool play) {}
bool RTSPClientSrc::handleCommand(Command::CommandType type, frame_sp& frame) 
{
    if (type == Command::CommandType::Relay)
	{
        return Module::handleCommand(type, frame);
    }
    return true; 
}

int RTSPClientSrc::getCurrentFps()
{
    return mDetail->currentCameraFps;
}

bool RTSPClientSrc::handlePropsChange(frame_sp& frame) { return true; }
