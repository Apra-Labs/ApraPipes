#include <cstdint>

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
}

#include "AbsControlModule.h"
#include "Frame.h"
#include "H264FrameDemuxer.h"
#include "H264ParserUtils.h"
#include "H264Utils.h"
#include "RTSPPusher.h"
class RTSPPusher::Detail
{
	/* video output */
	AVFrame *frame;
	AVPicture src_picture, dst_picture;
	AVFormatContext *outContext;
	AVStream *video_st;
	AVCodec *video_codec;
	string mURL;
	string mTitle;
	int64_t totalDuration;
	AVRational in_time_base;
	bool isTCP;
	uint32_t encoderTargetKbps;

	AVStream *add_stream(AVFormatContext *oc, AVCodec **codec, enum AVCodecID codec_id, int num, int den)
	{
		LOG_TRACE << "add_stream enter";
		AVStream *st = 0;

		/* find the encoder */
		*codec = avcodec_find_encoder(codec_id);
		if (!(*codec))
		{
			LOG_ERROR << "Could not find encoder for ^" << avcodec_get_name(codec_id) << "^" << "\n";
		}
		else
		{
			st = avformat_new_stream(oc, *codec);
			if (!st)
			{
				LOG_ERROR << "Could not allocate stream." << "\n";
			}
			else
			{
				st->id = oc->nb_streams - 1;
				st->time_base.den = /*st->pts.den = */ 90000;
				st->time_base.num = /*st->pts.num = */ 1;

				in_time_base.den = den;
				in_time_base.num = num;
				auto c = st->codecpar;
				c->codec_id = codec_id;
				c->codec_type = AVMEDIA_TYPE_VIDEO;
				c->bit_rate = static_cast<int64_t>(encoderTargetKbps * 1024);
				c->width = (int)width;
				c->height = (int)height;
				c->format = AV_PIX_FMT_YUV420P;
			}
		}
		LOG_TRACE << "add_stream exit";
		return st;
	}

	int open_video_precoded()
	{
		AVCodecParameters *c = video_st->codecpar;

		c->extradata = (uint8_t *)(demuxer->getSPS_PPS().data());
		c->extradata_size = (int)demuxer->getSPS_PPS().size();
		lastDiff = pts_adder = lastPTS = 0;
		return 0;
	}

	bool init_stream_params()
	{
		LOG_TRACE << "init_stream_params";
		const_buffer sps_pps = demuxer->getSPS_PPS();
		sps_pps_properties p;
		H264ParserUtils::parse_sps(((const char *)sps_pps.data()) + 5, sps_pps.size() > 5 ? sps_pps.size() - 5 : sps_pps.size(), &p);
		width = p.width;
		height = p.height;
		bitrate = 2000000;
		fps_den = 30;
		fps_num = 1;

		return true;
	}

public:
	// We need to pass object reference to refer to the connection status etc
	// bool write_precoded_video_frame(boost::shared_ptr<Frame>& f, RTSPPusher&
	// rtspMod)
	bool write_precoded_video_frame(boost::shared_ptr<Frame> &f)
	{
		mutable_buffer &codedFrame = *(f.get());
		bool isKeyFrame = ((f->mFrameType == H264Utils::H264_NAL_TYPE_IDR_SLICE) || (f->mFrameType == H264Utils::H264_NAL_TYPE_SEQ_PARAM));
		totalDuration += duration;

		pkt->stream_index = video_st->index;
		pkt->pts = totalDuration;
		pkt->dts = pkt->pts;

		pkt->data = (uint8_t *)codedFrame.data();
		pkt->size = (int)codedFrame.size();
		if (isKeyFrame)
			pkt->flags |= AV_PKT_FLAG_KEY;

		int ret = av_write_frame(outContext, pkt);

		bool bRC = true;
		if (ret < 0)
		{
			bRC = false;

			char avErrorBuf[500] = {'\0'};
			av_strerror(ret, avErrorBuf, 500);

			LOG_ERROR << "Error while writing video frame : " << ret << ":"
					  << avErrorBuf << ":" << pkt->pts << "\n";

			// On evostream going down the return code is -32 - errno.h says 32 is
			// EPIPE but AVERROR_EOF not coming as -32

			if (ret == -32) // Need to resolve the corresponding enum in AVERROR
							// header files etc
			{
				connectionStatus = CONNECTION_FAILED;
				// emit the event after returning
			}
		}
		return bRC;
	}
	size_t width, height, bitrate;
	size_t fps_den = 30;
	size_t fps_num = 1;
	int64_t lastPTS, lastDiff, pts_adder, duration = 0;
	boost::shared_ptr<H264FrameDemuxer> demuxer;
	EventType connectionStatus;
	bool isFirstFrame;
	AVPacket *pkt;

	Detail(RTSPPusherProps props) : mURL(props.URL), mTitle(props.title), isTCP(props.isTCP),
									connectionStatus(CONNECTION_FAILED), isFirstFrame(false), duration(0),
									encoderTargetKbps(props.encoderTargetKbps)
	{
		demuxer = boost::shared_ptr<H264FrameDemuxer>(new H264FrameDemuxer());
		pkt = av_packet_alloc();
	}
	~Detail() { av_packet_free(&pkt); }

	bool init()
	{
		const char *url = mURL.c_str();

		int ret = 0;
		totalDuration = 0;

		av_log_set_level(AV_LOG_INFO);

		avformat_network_init();

		int rc = avformat_alloc_output_context2(&outContext, NULL, "rtsp", url);

		if (rc < 0)
		{
			LOG_FATAL << "Alloc failure in avformat : " << rc << "<>" << url;
			{
				char errBuf[AV_ERROR_MAX_STRING_SIZE];
				size_t errBufSize = AV_ERROR_MAX_STRING_SIZE;
				av_strerror(rc, errBuf, errBufSize);

				LOG_ERROR << errBuf;
			}
			return false;
		}

		if (!outContext)
		{
			LOG_FATAL << "Could not allocate an output context for : " << "^" << url << "^" << "\n";
			return false;
		}

		if (!outContext->oformat)
		{
			LOG_FATAL << "Could not create the output format for : " << "^" << url << "^" << "\n";
			return false;
		}
		return true;
	}

	bool write_header(int num, int den)
	{
		init_stream_params();
		int ret = 0;
		video_st = add_stream(outContext, &video_codec, AV_CODEC_ID_H264, num, den);

		/* Now that all the parameters are set, we can open the video codec and
		 * allocate the necessary encode buffers. */
		if (video_st)
		{
			LOG_INFO << "Video stream codec : ^" << avcodec_get_name(video_st->codecpar->codec_id) << "^" << "\n";

			ret = open_video_precoded();
			if (ret < 0)
			{
				LOG_FATAL << "Open video stream failed." << "\n";
				return false;
			}
		}
		else
		{
			LOG_FATAL << "Add video stream for the codec : ^" << avcodec_get_name(AV_CODEC_ID_H264) << "^ " << "failed" << "\n";
			return false;
		}

		AVDictionary *options = NULL;

		av_dict_set(&outContext->metadata, "title", mTitle.c_str(), 0);
		av_dict_set(&options, "rtsp_transport", isTCP ? "tcp" : "udp", 0);

		av_dump_format(outContext, 0, mURL.c_str(), 1);

		AVFormatContext *ac[] = {outContext};
		char buf[1024];
		av_sdp_create(ac, 1, buf, 1024);

		ret = avformat_write_header(outContext, &options);

		// Why is it not returning if RTSP server is not available?
		// is there some timeout mechanism that needs to be implemented?
		if (ret != 0)
		{

			LOG_ERROR << "Failed to connect to RTSP server for ^" << mURL.c_str() << "^" << "\n";
			return false;
		}

		if (!duration)
		{
			calculateDuration();
		}

		return true;
	}

	void calculateDuration(int den = 0)
	{
		if (den)
		{
			in_time_base.den = den;
			fps_den = den;
		}
		else
		{
			in_time_base.den = fps_den;
		}
		AVRational timeBase;
		timeBase.den = 90000;
		timeBase.num = 1;
		in_time_base.num = fps_num;
		duration = av_rescale_q_rnd(1, in_time_base, timeBase, AVRounding(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
	}

	bool term(EventType status)
	{
		if (video_st)
		{
			if (status != CONNECTION_FAILED)
			{
				av_write_trailer(outContext);
			}

			video_st->codecpar->extradata = 0;
			video_st->codecpar->extradata_size = 0;
		}

		if (outContext)
		{
			avformat_free_context(outContext);
			return true;
		}
		else
		{
			return false;
		}

		// if connection has not been established, free context is causing an
		// an unhandled exception  because of heap correuption
		// if we free outContext without freeing up the other members of video_st
	}
};

RTSPPusher::RTSPPusher(RTSPPusherProps props) : Module(SINK, "RTSPPusher", props), pausedState(false)
{
	mDetail.reset(new RTSPPusher::Detail(props));

	// handles the frame drops and initial parsing
	adaptQueue(mDetail->demuxer);
	paceMaker = boost::shared_ptr<PaceMaker>(new PaceMaker(props.fps));
}

RTSPPusher::~RTSPPusher() { mDetail.reset(); }

bool RTSPPusher::init()
{
	if (!Module::init())
	{
		return false;
	}

	return mDetail->init();
}

bool RTSPPusher::term()
{
	bool bRC = mDetail->term(mDetail->connectionStatus);

	auto res = Module::term();

	return bRC && res;
}

bool RTSPPusher::validateInputPins()
{
	framemetadata_sp metadata = getFirstInputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::H264_DATA)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be "
									   "H264_DATA. Actual<"
				  << frameType << ">";
		return false;
	}

	if (metadata->getMemType() != FrameMetadata::HOST)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input MemType is expected to be HOST. Actual<" << metadata->getMemType() << ">";
		return false;
	}

	return true;
}

bool RTSPPusher::process(frame_container &frames)
{
	auto frame = frames.begin()->second;
	bool isKeyFrame = (frame->mFrameType == H264Utils::H264_NAL_TYPE_IDR_SLICE || frame->mFrameType == H264Utils::H264_NAL_TYPE_SEQ_PARAM);

	if (isKeyFrame)
	{
		savedIFrame = frame;
	}

	if (mDetail->connectionStatus != CONNECTION_READY)
	{
		return true;
	}

	if (!pausedState)
	{
		if (!mDetail->write_precoded_video_frame(frame))
		{
			mDetail->connectionStatus = WRITE_FAILED;
			LOG_FATAL << "write_precoded_video_frame failed";
			return false;
		}
		else
		{
			LOG_TRACE << "write_precoded_video_frame successful";
		}
	}
	return true;
}

bool RTSPPusher::setPausedState(bool state)
{
	LOG_TRACE << "RTSP-PUSHER: I am in setPausedState function with state - " << state;
	pausedState = state;
	if (state)
	{
		if (controlModule != nullptr)
		{
			boost::shared_ptr<AbsControlModule> ctl = boost::dynamic_pointer_cast<AbsControlModule>(controlModule);
			ctl->handlePusherPauseTS(savedIFrame->timestamp);
		}
		if (!pauserThread.joinable())
		{
			pauserThread = boost::thread(&RTSPPusher::pauserThreadFunction, this);
		}
	}
	else
	{
		setFps(fps);
		if (pauserThread.joinable())
		{
			pauserThread.interrupt();
			pauserThread.join();
		}
	}
	return true;
}

void RTSPPusher::pauserThreadFunction()
{
	try
	{
		auto sendFrame = savedIFrame;
		while (pausedState)
		{
			paceMaker->start();
			std::chrono::time_point<std::chrono::system_clock> t = std::chrono::system_clock::now();
			auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(t.time_since_epoch());

			if (!mDetail)
			{
				LOG_ERROR << "mDetail is null";
				break;
			}

			if (!mDetail->write_precoded_video_frame(sendFrame))
			{
				mDetail->connectionStatus = WRITE_FAILED;
				LOG_FATAL << "write_precoded_video_frame failed";
			}
			std::chrono::time_point<std::chrono::system_clock> t1 = std::chrono::system_clock::now();
			auto dur1 = std::chrono::duration_cast<std::chrono::milliseconds>(t1.time_since_epoch());
			auto diff = dur1.count() - dur.count();
			// int newfps = 1000 / diff;
			int newfps = 20;
			paceMaker->end();
			setFps(newfps, true);
		}
	}
	catch (boost::thread_interrupted &)
	{
		LOG_INFO << "Pause thread interrupted. Going to PlayState...";
	}
}

void RTSPPusher::setFps(int _fps, bool pausedState)
{
	if (!pausedState)
	{
		fps = _fps;
	}
	paceMaker->setFps(_fps);
	mDetail->calculateDuration(_fps);
}

bool RTSPPusher::processSOS(frame_sp &frame)
{
	LOG_INFO << "processSoS() called";
	// stick the sps/pps into extradata
	if (mDetail->write_header(1, 30) && mDetail->write_precoded_video_frame(frame))
	{
		// written header and first frame both.
		mDetail->connectionStatus = CONNECTION_READY;
		mDetail->isFirstFrame = true;

		return true;
	}
	else
	{
		LOG_ERROR << "Could not write stream header... stream will not play !!" << "\n";
		mDetail->connectionStatus = CONNECTION_FAILED;

		return false;
	}

	return true;
}

bool RTSPPusher::shouldTriggerSOS()
{
	std::string err = mDetail->connectionStatus == CONNECTION_FAILED ? "connection failed" : "stream ended";
	LOG_DEBUG << "shouldTriggerSOS connvetion status<" << err << ">";

	return mDetail->connectionStatus == CONNECTION_FAILED || mDetail->connectionStatus == STREAM_ENDED;
}

bool RTSPPusher::processEOS(string &pinId)
{
	LOG_INFO << "EOS recieved for pinID <" << pinId << ">";
	// mDetail->connectionStatus = STREAM_ENDED;
	return true;
}
