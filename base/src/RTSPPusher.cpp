#include <cstdint>

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
}

#include "RTSPPusher.h"
#include "H264FrameDemuxer.h"
#include "H264Utils.h"
#include "H264ParserUtils.h"
#include "Frame.h"

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
			// av_log(NULL, AV_LOG_ERROR, "Could not find encoder for '%s'.\n", avcodec_get_name(codec_id));
			LOG_ERROR << "Could not find encoder for ^" << avcodec_get_name(codec_id) << "^"
					  << "\n";
		}
		else
		{
			st = avformat_new_stream(oc, *codec);
			if (!st)
			{
				// av_log(NULL, AV_LOG_ERROR, "Could not allocate stream.\n");
				LOG_ERROR << "Could not allocate stream."
						  << "\n";
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
				c->bit_rate = static_cast<int64_t>(encoderTargetKbps*1024);
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
		fps_den = 1;
		fps_num = 30;

		return true;
	}

public:
	// We need to pass object reference to refer to the connection status etc
	// bool write_precoded_video_frame(boost::shared_ptr<Frame>& f, RTSPPusher& rtspMod)
	bool write_precoded_video_frame(boost::shared_ptr<Frame> &f)
	{
		mutable_buffer &codedFrame = *(f.get());
		bool isKeyFrame = (f->mFrameType == H264Utils::H264_NAL_TYPE_IDR_SLICE);

		AVPacket pkt = {0};
		av_init_packet(&pkt);

		/* encode the image */

		pkt.stream_index = video_st->index;

		totalDuration += duration;
		pkt.pts = totalDuration;
		pkt.dts = pkt.pts;

		pkt.data = (uint8_t *)codedFrame.data();
		pkt.size = (int)codedFrame.size();
		if (isKeyFrame)
			pkt.flags |= AV_PKT_FLAG_KEY;

		int ret = av_write_frame(outContext, &pkt);

		if (ret < 0)
		{
			// av_log(NULL, AV_LOG_ERROR, "Error while writing video frame.ret = %d\n", ret);

			char avErrorBuf[500] = {'\0'};
			av_strerror(ret, avErrorBuf, 500);

			LOG_ERROR << "Error while writing video frame : " << ret << ":" << avErrorBuf << ":" << pkt.pts << "\n";

			// On evostream going down the return code is -32 - errno.h says 32 is EPIPE but
			// AVERROR_EOF not coming as -32

			if (ret == -32) // Need to resolve the corresponding enum in AVERROR header files etc
			{
				connectionStatus = CONNECTION_FAILED;
				// emit the event after returning
			}

			return false;
		}
		return true;
	}
	size_t width, height, bitrate, fps_den, fps_num;
	int64_t lastPTS, lastDiff, pts_adder, duration;
	boost::shared_ptr<H264FrameDemuxer> demuxer;
	EventType connectionStatus;
	bool isFirstFrame;

	Detail(RTSPPusherProps props) : mURL(props.URL), mTitle(props.title), isTCP(props.isTCP), connectionStatus(CONNECTION_FAILED), isFirstFrame(false), duration(0), encoderTargetKbps(props.encoderTargetKbps)
	{
		demuxer = boost::shared_ptr<H264FrameDemuxer>(new H264FrameDemuxer());
	}
	~Detail()
	{
	}

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
			// av_log(NULL, AV_LOG_FATAL, "Alloc failure in avformat %d\n", rc);
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
			// av_log(NULL, AV_LOG_FATAL, "Could not allocate an output context for '%s'.\n", url);
			LOG_FATAL << "Could not allocate an output context for : "
					  << "^" << url << "^"
					  << "\n";
			return false;
		}

		if (!outContext->oformat)
		{
			// av_log(NULL, AV_LOG_FATAL, "Could not create the output format for '%s'.\n", url);
			LOG_FATAL << "Could not create the output format for : "
					  << "^" << url << "^"
					  << "\n";
			return false;
		}
		return true;
	}

	bool write_header(int num, int den)
	{
		init_stream_params();
		int ret = 0;
		video_st = add_stream(outContext, &video_codec, AV_CODEC_ID_H264, num, den);

		/* Now that all the parameters are set, we can open the video codec and allocate the necessary encode buffers. */
		if (video_st)
		{
			LOG_INFO << "Video stream codec : ^" << avcodec_get_name(video_st->codecpar->codec_id) << "^"
					 << "\n";

			ret = open_video_precoded();
			if (ret < 0)
			{
				// av_log(NULL, AV_LOG_FATAL, "Open video stream failed.\n");
				LOG_FATAL << "Open video stream failed."
						  << "\n";
				return false;
			}
		}
		else
		{
			LOG_FATAL << "Add video stream for the codec : ^" << avcodec_get_name(AV_CODEC_ID_H264) << "^ "
					  << "failed"
					  << "\n";
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
			// av_log(NULL, AV_LOG_ERROR, "Failed to connect to RTSP server for '%s'.\n", mURL.c_str());

			LOG_ERROR << "Failed to connect to RTSP server for ^" << mURL.c_str() << "^"
					  << "\n";
			return false;
		}

		duration = av_rescale_q_rnd(1, in_time_base, video_st->time_base, AVRounding(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));

		return true;
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

RTSPPusher::RTSPPusher(RTSPPusherProps props) : Module(SINK, "RTSPPusher", props)
{
	mDetail.reset(new RTSPPusher::Detail(props));

	//handles the frame drops and initial parsing
	adaptQueue(mDetail->demuxer);
}

RTSPPusher::~RTSPPusher()
{
	mDetail.reset();
}

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
	if (mDetail->connectionStatus == WRITE_FAILED || mDetail->connectionStatus == STREAM_ENDED)
	{
		// self destruct
		// emit_fatal(mDetail->connectionStatus);
	}

	auto res = Module::term();

	return bRC && res;
}

bool RTSPPusher::validateInputPins()
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

	if (mDetail->connectionStatus != CONNECTION_READY)
	{
		return true;
	}

	if (mDetail->isFirstFrame)
	{
		mDetail->isFirstFrame = false;
		return true;
	}

	// non-first frame
	if (!mDetail->write_precoded_video_frame(frame))
	{
		mDetail->connectionStatus = WRITE_FAILED;
		LOG_FATAL << "write_precoded_video_frame failed";

		return false;
	}

	return true;
}

bool RTSPPusher::processSOS(frame_sp &frame)
{
	LOG_TRACE << "at first frame";
	//stick the sps/pps into extradata
	if (mDetail->write_header(1, 30) && mDetail->write_precoded_video_frame(frame))
	{
		//written header and first frame both.
		mDetail->connectionStatus = CONNECTION_READY;
		mDetail->isFirstFrame = true;
		// emit_event(CONNECTION_READY);
		return true;
	}
	else
	{
		LOG_ERROR << "Could not write stream header... stream will not play !!"
				  << "\n";
		mDetail->connectionStatus = CONNECTION_FAILED;
		// emit_event(CONNECTION_FAILED);
		return false;
	}

	return true;
}

bool RTSPPusher::shouldTriggerSOS()
{
	return mDetail->connectionStatus == CONNECTION_FAILED || mDetail->connectionStatus == STREAM_ENDED;
}

bool RTSPPusher::processEOS(string &pinId)
{
	mDetail->connectionStatus = STREAM_ENDED;

	return true;
}
