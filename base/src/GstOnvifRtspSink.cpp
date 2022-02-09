#include "GstOnvifRtspSink.h"
#include "FrameMetadata.h"
#include "H264Metadata.h"

#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "AIPExceptions.h"

#include <gst/gst.h>
#include <gst/app/app.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <stdio.h>
#include <string.h>

void media_configure(GstRTSPMediaFactory *factory, GstRTSPMedia *media, gpointer user_data);

void start_feed(GstElement *pipeline, guint size, gpointer user_data);

void read_data(void *user_data);

void stop_feed(GstElement *pipeline, void *user_data);

struct GSTContext
{
public:
	GstElement *element;
	GstElement *appsrc;
	guint sourceid;
	GstClockTime timestamp;
};

class GStreamerOnvifRTSPSink::Detail
{
public:
	Detail(GStreamerOnvifRTSPSinkProps &_props, std::function<void()> _step) : mProps(_props), mStep(_step), ctx(nullptr)
	{
	}

	~Detail()
	{
	}

	void setMetadata(framemetadata_sp &metadata)
	{
		frameType = metadata->getFrameType();
		if (frameType == FrameMetadata::H264_DATA)
		{
			auto inputH264Metadata = FrameMetadataFactory::downcast<H264Metadata>(metadata);
			width = inputH264Metadata->getWidth();
			height = inputH264Metadata->getHeight();
		}
		else if (frameType == FrameMetadata::RAW_IMAGE_PLANAR)
		{
			auto inputRawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
			width = mProps.width;
			height = mProps.height;
		}
		else if (frameType == FrameMetadata::RAW_IMAGE)
		{
			auto inputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
			width = inputRawMetadata->getWidth();
			height = inputRawMetadata->getHeight();
		}
	}

	void setProps(GStreamerOnvifRTSPSinkProps &props)
	{
		mProps = props;
	}

	GStreamerOnvifRTSPSinkProps getProps()
	{
		return mProps;
	}

	gboolean read_data_actual(GSTContext *ctx)
	{
		GstMapInfo map;
		GstFlowReturn gstret;
		buffer = gst_buffer_new_allocate(NULL, frame->size(), NULL);
		gst_buffer_map(buffer, &map, GST_MAP_WRITE);

		map.size = frame->size();
		memcpy(map.data, frame->data(), frame->size());
		GST_BUFFER_PTS(buffer) = ctx->timestamp;
		GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale_int(1, GST_SECOND, mProps.fps);
		ctx->timestamp += GST_BUFFER_DURATION(buffer);

		if (frame->size() > 0)
		{
			gstret = gst_app_src_push_buffer((GstAppSrc *)ctx->appsrc, buffer);
			if (gstret != GST_FLOW_OK)
			{
				g_source_remove(ctx->sourceid);
				ctx->sourceid = 0;
				return false;
			}
		}
		else
		{
			printf("\n failed to read\n");
			return false;
		}

		gst_buffer_unmap(buffer, &map);

		return true;
	}

	void media_configure_helper(GstRTSPMediaFactory *factory, GstRTSPMedia *media)
	{

		ctx = g_new0(GSTContext, 1);

		ctx->element = gst_rtsp_media_get_element(media);

		ctx->appsrc = gst_bin_get_by_name_recurse_up(GST_BIN(ctx->element),
													 "gstsrc");

		gst_util_set_object_arg(G_OBJECT(ctx->appsrc), "format", "time");
		g_object_set(G_OBJECT(ctx->appsrc), "format", GST_FORMAT_TIME, NULL);

		if (frameType == FrameMetadata::H264_DATA)
		{
			g_object_set(G_OBJECT(ctx->appsrc), "caps",
						 gst_caps_new_simple("video/x-h264",
											 "format", G_TYPE_STRING, "I420",
											 "width", G_TYPE_INT, (gint)mProps.width,
											 "height", G_TYPE_INT, (gint)mProps.height,
											 "framerate", GST_TYPE_FRACTION, 0, 1, NULL),
						 NULL);
		}
		else if (frameType == FrameMetadata::RAW_IMAGE_PLANAR)
		{
			g_object_set(G_OBJECT(ctx->appsrc), "caps",
						 gst_caps_new_simple("video/x-raw",
											 "format", G_TYPE_STRING, "I420",
											 "width", G_TYPE_INT, (gint)mProps.width,
											 "height", G_TYPE_INT, (gint)mProps.height,
											 "framerate", GST_TYPE_FRACTION, 0, 1, NULL),
						 NULL);
		}

		else if (frameType == FrameMetadata::RAW_IMAGE)
		{
			g_object_set(G_OBJECT(ctx->appsrc), "caps",
						 gst_caps_new_simple("video/x-raw",
											 "format", G_TYPE_STRING, "RGB",
											 "width", G_TYPE_INT, (gint)mProps.width,
											 "height", G_TYPE_INT, (gint)mProps.height,
											 "framerate", GST_TYPE_FRACTION, mProps.fps, 1, NULL),
						 NULL);
		}

		g_object_set_data_full(G_OBJECT(media), "my-extra-data", ctx,
							   (GDestroyNotify)g_free);

		g_signal_connect(ctx->appsrc, "need-data", G_CALLBACK(start_feed), this);
		g_signal_connect(ctx->appsrc, "enough-data", G_CALLBACK(stop_feed), this);
	}

	bool gstreamerMainThread(GMainLoop *loop)
	{

		gst_init(0, NULL);

		loop = g_main_loop_new(NULL, FALSE);

		server = gst_rtsp_onvif_server_new();

		mounts = gst_rtsp_server_get_mount_points(server);

		factory = gst_rtsp_media_factory_new();

		if (frameType == FrameMetadata::H264_DATA)
		{
			gst_rtsp_media_factory_set_launch(factory, "(appsrc is-live=TRUE block=TRUE name=gstsrc min-latency=0 ! video/x-h264,stream-format=byte-stream ! queue ! h264parse ! rtph264pay name=pay0 pt=96 )");
		}

		else if (frameType == FrameMetadata::RAW_IMAGE_PLANAR)
		{
			sprintf(pipelineDescription, "(appsrc is-live=TRUE name=gstsrc ! x264enc tune=zerolatency speed-preset=superfast bitrate=%d key-int-max=%d ! video/x-h264, stream-format=byte-stream , profile=%s ! queue ! h264parse ! rtph264pay name=pay0 pt=96 )", mProps.bitrate, mProps.goplength, mProps.h264Profile.c_str());
			gst_rtsp_media_factory_set_launch(factory, pipelineDescription);
		}

		else if (frameType == FrameMetadata::RAW_IMAGE)
		{
			sprintf(pipelineDescription, "(appsrc is-live=TRUE block=TRUE name=gstsrc min-latency=0 ! videoconvert ! video/x-raw, format=I420 ! x264enc tune=zerolatency speed-preset=superfast bitrate=%d key-int-max=%d ! video/x-h264, stream-format=byte-stream, profile=%s ! rtph264pay name=pay0 pt=96 )", mProps.bitrate, mProps.goplength, mProps.h264Profile.c_str());
			gst_rtsp_media_factory_set_launch(factory, pipelineDescription);
		}

		g_signal_connect(factory, "media-configure", (GCallback)media_configure, this);

		gst_rtsp_mount_points_add_factory(mounts, mProps.mountPoint.c_str(), factory);

		gst_rtsp_media_factory_set_shared(factory, true);

		gst_rtsp_media_factory_add_role(factory, "user",
										GST_RTSP_PERM_MEDIA_FACTORY_ACCESS, G_TYPE_BOOLEAN, TRUE,
										GST_RTSP_PERM_MEDIA_FACTORY_CONSTRUCT, G_TYPE_BOOLEAN, TRUE, NULL);

		g_object_unref(mounts);

		auth = gst_rtsp_auth_new();

		gst_rtsp_auth_set_supported_methods(auth, GST_RTSP_AUTH_DIGEST);

		if (!mProps.htdigestPath.empty())
		{
			token =
				gst_rtsp_token_new(GST_RTSP_TOKEN_MEDIA_FACTORY_ROLE, G_TYPE_STRING,
								   "user", NULL);

			if (!gst_rtsp_auth_parse_htdigest(auth, mProps.htdigestPath.c_str(), token))
			{
				g_printerr("Could not parse htdigest at %s\n", mProps.htdigestPath.c_str());
				gst_rtsp_token_unref(token);
				goto failed;
			}

			gst_rtsp_token_unref(token);
		}

		if (!mProps.realm.empty())
			gst_rtsp_auth_set_realm(auth, mProps.realm.c_str());

		gst_rtsp_server_set_auth(server, auth);
		gst_rtsp_server_set_address(server, mProps.unicastAddress.c_str());
		gst_rtsp_server_set_service(server, mProps.port.c_str());
		g_object_unref(auth);

		if (gst_rtsp_server_attach(server, NULL) == 0)
			goto failed;

		if (!mProps.htdigestPath.empty())
			g_print("stream with htdigest users ready at rtsp://%s:%d%s\n", mProps.unicastAddress.c_str(), gst_rtsp_server_get_bound_port(server), mProps.mountPoint.c_str());

		g_main_loop_run(loop);

	failed:
	{
		g_print("failed to attach the server\n");
		return false;
	}

		return true;
	}

public:
	GStreamerOnvifRTSPSinkProps mProps;
	GstBuffer *buffer;
	GSTContext *ctx;
	GMainLoop *loop;
	GstRTSPServer *server;
	GstRTSPMountPoints *mounts;
	GstRTSPMediaFactory *factory;
	GstRTSPAuth *auth;
	GstRTSPToken *token;
	frame_sp frame;
	int width = 0;
	int height;
	char pipelineDescription[10000];
	FrameMetadata::FrameType frameType;
	std::function<void()> mStep;
	std::thread gstThread;
};

void media_configure(GstRTSPMediaFactory *factory, GstRTSPMedia *media, gpointer user_data)
{
	auto detail = static_cast<GStreamerOnvifRTSPSink::Detail *>(user_data);
	detail->media_configure_helper(factory, media);
	return;
}

void start_feed(GstElement *pipeline, guint size, void *user_data)
{
	auto detail = static_cast<GStreamerOnvifRTSPSink::Detail *>(user_data);
	if (detail->ctx->sourceid == 0)
	{
		detail->ctx->sourceid = g_idle_add((GSourceFunc)read_data, user_data);
	}
	return;
}

void read_data(void *user_data)
{
	auto detail = static_cast<GStreamerOnvifRTSPSink::Detail *>(user_data);
	detail->mStep();
	return;
}

void stop_feed(GstElement *pipeline, void *user_data)
{
	auto detail = static_cast<GStreamerOnvifRTSPSink::Detail *>(user_data);
	if (detail->ctx->sourceid != 0)
	{
		g_source_remove(detail->ctx->sourceid);
		detail->ctx->sourceid = 0;
	}
	return;
}

GStreamerOnvifRTSPSink::GStreamerOnvifRTSPSink(GStreamerOnvifRTSPSinkProps props) : Module(SINK, "GStreamerOnvifRTSPSink", props)
{
	mDetail.reset(new Detail(props, [&]() -> void
							 { Module::step(); }));
}

GStreamerOnvifRTSPSink::~GStreamerOnvifRTSPSink() {}

bool GStreamerOnvifRTSPSink::init()
{
	if (!Module::init())
	{
		return false;
	}

	framemetadata_sp metadata = getFirstInputMetadata();
	mDetail->setMetadata(metadata);

	mDetail->gstThread = std::thread(&GStreamerOnvifRTSPSink::Detail::gstreamerMainThread, mDetail.get(), mDetail->loop);

	return true;
}

bool GStreamerOnvifRTSPSink::term()
{

	g_main_loop_quit(mDetail->loop);
	mDetail->gstThread.join();
	gst_deinit();
	return Module::term();
}

bool GStreamerOnvifRTSPSink::validateInputPins()
{
	if (getNumberOfInputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstInputMetadata();

	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::H264_DATA && frameType != FrameMetadata::RAW_IMAGE_PLANAR && frameType != FrameMetadata::RAW_IMAGE)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be H264 or RAW or RAW_PLANAR. Actual<" << frameType << ">";
		return false;
	}
	return true;
}

bool GStreamerOnvifRTSPSink::process(frame_container &frames)
{
	mDetail->frame = frames.cbegin()->second;
	mDetail->read_data_actual(mDetail->ctx);
	return true;
}

bool GStreamerOnvifRTSPSink::processSOS(frame_sp &frame)
{
	return true;
}

bool GStreamerOnvifRTSPSink::shouldTriggerSOS()
{
	return mDetail->width == 0;
}

bool GStreamerOnvifRTSPSink::processEOS(string &pinId)
{
	return false;
}

GStreamerOnvifRTSPSinkProps GStreamerOnvifRTSPSink::getProps()
{
	auto mProps = mDetail->getProps();
	fillProps(mProps);

	return mProps;
}

void GStreamerOnvifRTSPSink::setProps(GStreamerOnvifRTSPSinkProps &mProps)
{
	Module::addPropsToQueue(mProps);
}

bool GStreamerOnvifRTSPSink::handlePropsChange(frame_sp &frame)
{
	GStreamerOnvifRTSPSinkProps mProps;
	bool ret = Module::handlePropsChange(frame, mProps);
	mDetail->setProps(mProps);

	GstRTSPAuth *newauth = gst_rtsp_auth_new();
	gst_rtsp_auth_set_supported_methods(newauth, GST_RTSP_AUTH_DIGEST);
	gst_rtsp_auth_set_realm(newauth, mDetail->mProps.realm.c_str());

	GstRTSPToken *token = gst_rtsp_token_new(GST_RTSP_TOKEN_MEDIA_FACTORY_ROLE,
											 G_TYPE_STRING,
											 "user", NULL);

	for (const auto &user : mDetail->mProps.userList)
	{
		gst_rtsp_auth_add_digest(newauth, user.username.c_str(), user.password.c_str(), token);
		
	}
	gst_rtsp_token_unref(token);
	gst_rtsp_server_set_auth(mDetail->server, newauth);
	if (mDetail->auth)
		g_object_unref(mDetail->auth);
	mDetail->auth = newauth;

	sendEOS();

	return ret;
}
