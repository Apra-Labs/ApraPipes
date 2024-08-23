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
#include "AbsControlModule.h"

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
    Detail(RTSPClientSrc* m,std::string path, bool useTCP, int urlTimeout) :myModule(m), path(path), bConnected(false), bUseTCP(useTCP), iUrlTimeout(urlTimeout)
    {
		int_ctx = std::make_pair(iUrlTimeout, start_time);
		int_cb = { interrupt_cb, &int_ctx };
	}
    ~Detail() { destroy(); }
    void destroy()
    {
        if(nullptr!= pFormatCtx)
            avformat_close_input(&pFormatCtx);
    }

    // https://stackoverflow.com/questions/10666242/detecting-a-timeout-in-ffmpeg
    static int interrupt_cb(void *ctx)
    {
        std::pair<int, time_t>* context = static_cast<std::pair<int, time_t>*>(ctx);
        int timeout = context->first;
        time_t start_time = context->second;
        if ((time(NULL) - start_time) > timeout)
            return 1; // Return 1 to interrupt the operation
        return 0; 
    }

    bool connect()
    {
        avformat_network_init();
        av_register_all();

        //Start time initialize
		int_ctx.second = time(NULL);

        pFormatCtx = avformat_alloc_context();
		pFormatCtx->interrupt_callback = int_cb;

        AVDictionary* avdic = NULL;

        av_dict_set(&avdic, "rtsp_transport", (bUseTCP)?"tcp":"udp", 0);
        av_dict_set(&avdic, "max_delay", "100", 0);

        if (avformat_open_input(&pFormatCtx, path.c_str(), NULL, &avdic) != 0)
        {
            LOG_ERROR << "can't open the URL." << path <<std::endl;
            bConnected = false;
            return bConnected;
        }

		int_ctx.second = time(NULL);

        if (avformat_find_stream_info(pFormatCtx, NULL) < 0)
        {
            LOG_ERROR << "can't find stream infomation" << std::endl;
            bConnected = false;
            return bConnected;
        }
        LOG_INFO << "Opened url and found stream info";
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
        saveSpsPps();
        return bConnected;
    }

    void saveSpsPps()
    {
        H264Utils::extractSpsAndPpsFromExtradata((char*)pFormatCtx->streams[0]->codec->extradata, pFormatCtx->streams[0]->codec->extradata_size, spsData, spsSize, ppsData, ppsSize);
    }

    frame_sp prependSpsPpsToFrame(std::string id, size_t offsetToSkip, int naluSeparatorSize)
    {
        packet.data += (offsetToSkip - naluSeparatorSize);
        packet.size -= (offsetToSkip - naluSeparatorSize);
        size_t totalFrameSize = packet.size + spsSize + ppsSize;

        auto frm = myModule->makeFrame(totalFrameSize, id);
        uint8_t* frameData = static_cast<uint8_t*>(frm->data());
        memcpy(frameData, spsData, spsSize);
        frameData += spsSize;
        memcpy(frameData, ppsData, ppsSize);
        frameData += ppsSize;
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
        int count = 0;
        while(!got_something)
        {
			LOG_TRACE << "got something from stream ? <" << got_something << ">";

			int_ctx.second = time(NULL);

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
              
                size_t offset = 0;
                int naluSeparatorSize = 0;
                H264Utils::getNALUnitOffsetAndSizeBasedOnGivenType((char*)packet.data, packet.size, offset, naluSeparatorSize, H264Utils::H264_NAL_TYPE_IDR_SLICE);
                auto it = streamsMap.find(packet.stream_index);
                if (it != streamsMap.end()) { // so we have an interest in sending this
                    frame_sp frm;
                    if (!offset && !naluSeparatorSize)
                    {
                        frm = myModule->makeFrame(packet.size, it->second);
                        memcpy(frm->data(), packet.data, packet.size);
                    }
                    else
                    {

                        frm = prependSpsPpsToFrame(it->second, offset, naluSeparatorSize);
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
                        if (currentCameraFps && controlModule != nullptr && currentCameraFps != frameCount)
                        {
                            DecoderPlaybackSpeed cmd;
                            cmd.playbackSpeed = 1;
                            cmd.playbackFps = frameCount;
                            cmd.gop = 1;
                            bool priority = true;
                            boost::shared_ptr<AbsControlModule>ctl = boost::dynamic_pointer_cast<AbsControlModule>(controlModule);
                            ctl->handleDecoderSpeed(cmd, priority);
                        }
                        currentCameraFps = frameCount;
                        frameCount = 0;
                        beginTs = dur;
                    }
                    frameCount++;
                }
                av_packet_unref(&packet);
            }
            else
            {  //Inform control module about stopping of stream
                LOG_TRACE <<"Not getting data from source will retry - "<<count;
                count++;
                if(count == 10 && controlModule != nullptr)
                {
                    boost::shared_ptr<AbsControlModule> ctl = boost::dynamic_pointer_cast<AbsControlModule>(controlModule);
                    ctl->handleNoRTSPFrame(false);
                    return false;
                }
            }
        }

        if(outFrames.size()>0)
		{
			LOG_TRACE << "sending out data from RTSPCLIENTSRC !";
			myModule->send(outFrames);
		}
        return true;
    }

    bool isConncected() const { return bConnected; }
    int frameCount = 0;
    int currentCameraFps = 0;
    boost::shared_ptr<Module> controlModule = nullptr;
private:
    AVPacket packet;
    AVFormatContext* pFormatCtx = nullptr;
    std::string path;
    bool bConnected;
    int videoStream=-1;
    bool bUseTCP;
	int iUrlTimeout;
	time_t start_time;
    std::map<unsigned int, std::string> streamsMap;
    RTSPClientSrc* myModule;
	std::chrono::milliseconds beginTs;
	bool initDone = false;
    char* spsData = nullptr;
    char* ppsData = nullptr;
    int spsSize = 0;
    int ppsSize = 0;
    
    std::pair<int, time_t> int_ctx;
    AVIOInterruptCB int_cb;
};

RTSPClientSrc::RTSPClientSrc(RTSPClientSrcProps _props) : Module(SOURCE, "RTSPClientSrc", _props), mProps(_props)
{
    mDetail.reset(new Detail(this, mProps.rtspURL, mProps.useTCP, mProps.timeout));
}
RTSPClientSrc::~RTSPClientSrc() {
	LOG_INFO << "Destructor called for rtspclientsrc";
}
bool RTSPClientSrc::init() {
    if (mDetail->connect())
    {
        return Module::init();
    }
    mDetail->controlModule = controlModule;
    return false;
}
bool RTSPClientSrc::term() {
    LOG_INFO << "Term called for rtspclientsrc";
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
	LOG_TRACE << "Produce called: starting readBuffer for RTSPCLIENTSRC";
    auto ret = mDetail->readBuffer();
	LOG_INFO << "Produce called: finished readBuffer for RTSPCLIENTSRC";
	return ret;
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
