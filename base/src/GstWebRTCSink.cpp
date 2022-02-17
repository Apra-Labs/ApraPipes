#include "GstWebRTCSink.h"
#include "FrameMetadata.h"
#include "H264Metadata.h"

#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "AIPExceptions.h"

#include <gst/gst.h>
#include <gst/gstpromise.h>
#include <gst/app/app.h>

#include <gst/sdp/sdp.h>

#define GST_USE_UNSTABLE_API
#include <gst/webrtc/webrtc.h>

/* For signalling */
#include <libsoup/soup.h>
#include <json-glib/json-glib.h>

#include <stdio.h>
#include <string.h>

#ifndef __KMS_AGNOSTIC_CAPS_H__
#define __KMS_AGNOSTIC_CAPS_H__

#define KMS_AGNOSTIC_RAW_AUDIO_CAPS \
	"audio/x-raw;"

#define KMS_AGNOSTIC_RAW_VIDEO_CAPS \
	"video/x-raw;"

#define KMS_AGNOSTIC_RAW_CAPS   \
	KMS_AGNOSTIC_RAW_AUDIO_CAPS \
	KMS_AGNOSTIC_RAW_VIDEO_CAPS

#define KMS_AGNOSTIC_RTP_AUDIO_CAPS \
	"application/x-rtp,media=audio;"

#define KMS_AGNOSTIC_RTP_VIDEO_CAPS \
	"application/x-rtp,media=video;"

#define KMS_AGNOSTIC_RTP_CAPS   \
	KMS_AGNOSTIC_RTP_AUDIO_CAPS \
	KMS_AGNOSTIC_RTP_VIDEO_CAPS

#define KMS_AGNOSTIC_FORMATS_AUDIO_CAPS        \
	"audio/x-sbc;"                             \
	"audio/x-mulaw;"                           \
	"audio/x-flac;"                            \
	"audio/x-alaw;"                            \
	"audio/x-speex;"                           \
	"audio/x-ac3;"                             \
	"audio/x-alac;"                            \
	"audio/mpeg,mpegversion=1,layer=2;"        \
	"audio/x-nellymoser;"                      \
	"audio/x-gst_ff-sonic;"                    \
	"audio/x-gst_ff-sonicls;"                  \
	"audio/x-wma,wmaversion=1;"                \
	"audio/x-wma,wmaversion=2;"                \
	"audio/x-dpcm,layout=roq;"                 \
	"audio/x-adpcm,layout=adx;"                \
	"audio/x-adpcm,layout=g726;"               \
	"audio/x-adpcm,layout=quicktime;"          \
	"audio/x-adpcm,layout=dvi;"                \
	"audio/x-adpcm,layout=microsoft;"          \
	"audio/x-adpcm,layout=swf;"                \
	"audio/x-adpcm,layout=yamaha;"             \
	"audio/mpeg,mpegversion=4;"                \
	"audio/mpeg,mpegversion=1,layer=3;"        \
	"audio/x-celt;"                            \
	"audio/mpeg,mpegversion=[2, 4];"           \
	"audio/x-vorbis;"                          \
	"audio/x-opus;"                            \
	"audio/AMR,rate=[8000, 16000],channels=1;" \
	"audio/x-gsm;"

#define KMS_AGNOSTIC_NO_RTP_AUDIO_CAPS \
	KMS_AGNOSTIC_RAW_AUDIO_CAPS        \
	KMS_AGNOSTIC_FORMATS_AUDIO_CAPS

#define KMS_AGNOSTIC_AUDIO_CAPS    \
	KMS_AGNOSTIC_NO_RTP_AUDIO_CAPS \
	KMS_AGNOSTIC_RTP_AUDIO_CAPS

#define KMS_AGNOSTIC_FORMATS_VIDEO_CAPS           \
	"video/x-dirac;"                              \
	"image/png;"                                  \
	"image/jpeg;"                                 \
	"video/x-smoke;"                              \
	"video/x-asus,asusversion=1;"                 \
	"video/x-asus,asusversion=2;"                 \
	"image/bmp;"                                  \
	"video/x-dnxhd;"                              \
	"video/x-dv;"                                 \
	"video/x-ffv,ffvversion=1;"                   \
	"video/x-gst_ff-ffvhuff;"                     \
	"video/x-flash-screen;"                       \
	"video/x-flash-video,flvversion=1;"           \
	"video/x-h261;"                               \
	"video/x-h263,variant=itu,h263version=h263;"  \
	"video/x-h263,variant=itu,h263version=h263p;" \
	"video/x-huffyuv;"                            \
	"image/jpeg;"                                 \
	"image/jpeg;"                                 \
	"video/mpeg,mpegversion=1;"                   \
	"video/mpeg,mpegversion=2;"                   \
	"video/mpeg,mpegversion=4;"                   \
	"video/x-msmpeg,msmpegversion=41;"            \
	"video/x-msmpeg,msmpegversion=42;"            \
	"video/x-msmpeg,msmpegversion=43;"            \
	"video/x-gst_ff-pam;"                         \
	"image/pbm;"                                  \
	"video/x-gst_ff-pgm;"                         \
	"video/x-gst_ff-pgmyuv;"                      \
	"image/png;"                                  \
	"image/ppm;"                                  \
	"video/x-rle,layout=quicktime;"               \
	"video/x-gst_ff-roqvideo;"                    \
	"video/x-pn-realvideo,rmversion=1;"           \
	"video/x-pn-realvideo,rmversion=2;"           \
	"video/x-gst_ff-snow;"                        \
	"video/x-svq,svqversion=1;"                   \
	"video/x-wmv,wmvversion=1;"                   \
	"video/x-wmv,wmvversion=2;"                   \
	"video/x-gst_ff-zmbv;"                        \
	"video/x-theora;"                             \
	"video/x-h264;"                               \
	"video/x-gst_ff-libxvid;"                     \
	"video/x-h264;"                               \
	"video/x-xvid;"                               \
	"video/mpeg,mpegversion=[1, 2];"              \
	"video/x-theora;"                             \
	"video/x-vp8;"                                \
	"application/x-yuv4mpeg,y4mversion=2;"

#define KMS_AGNOSTIC_NO_RTP_VIDEO_CAPS \
	KMS_AGNOSTIC_RAW_VIDEO_CAPS        \
	KMS_AGNOSTIC_FORMATS_VIDEO_CAPS

#define KMS_AGNOSTIC_VIDEO_CAPS    \
	KMS_AGNOSTIC_NO_RTP_VIDEO_CAPS \
	KMS_AGNOSTIC_RTP_VIDEO_CAPS

#define KMS_AGNOSTIC_DATA_CAPS \
	"application/data;"

#define KMS_AGNOSTIC_CAPS   \
	KMS_AGNOSTIC_AUDIO_CAPS \
	KMS_AGNOSTIC_VIDEO_CAPS

#define KMS_AGNOSTIC_NO_RTP_CAPS   \
	KMS_AGNOSTIC_NO_RTP_AUDIO_CAPS \
	KMS_AGNOSTIC_NO_RTP_VIDEO_CAPS

#endif /* __KMS_AGNOSTIC_CAPS_H__ */

#define STUN_SERVER " stun-server=stun://stun.l.google.com:19302 "
#define RTP_CAPS_OPUS "application/x-rtp,media=audio,encoding-name=OPUS,payload="
#define RTP_CAPS_VP8 "application/x-rtp,media=video,encoding-name=VP8,payload="

enum AppState
{
	APP_STATE_UNKNOWN = 0,
	APP_STATE_ERROR = 1, /* generic error */
	SERVER_CONNECTING = 1000,
	SERVER_CONNECTION_ERROR,
	SERVER_CONNECTED, /* Ready to register */
	SERVER_REGISTERING = 2000,
	SERVER_REGISTRATION_ERROR,
	SERVER_REGISTERED, /* Ready to call a peer */
	SERVER_CLOSED,	   /* server connection closed by us or the server */
	PEER_CONNECTING = 3000,
	PEER_CONNECTION_ERROR,
	PEER_CONNECTED,
	PEER_CALL_NEGOTIATING = 4000,
	PEER_CALL_STARTED,
	PEER_CALL_STOPPING,
	PEER_CALL_STOPPED,
	PEER_CALL_ERROR,
};

void startFeed(GstElement *pipeline, guint size, gpointer user_data);

void readData(void *user_data);

void on_incoming_decodebin_stream(GstElement *decodebin, GstPad *pad,
								  void *userdata);

void uridecodebin_element_added(GstBin *bin,
								GstElement *element, gpointer data);

void on_negotiation_needed(GstElement *element, gpointer userdata);

void send_ice_candidate_message(GstElement *webrtc G_GNUC_UNUSED, guint mlineindex,
								gchar *candidate, gpointer userdata);

void on_server_closed(SoupWebsocketConnection *conn G_GNUC_UNUSED,
					  gpointer userdata);

void on_server_connected(SoupSession *session, GAsyncResult *res,
						 gpointer userdata);

void on_server_message(SoupWebsocketConnection *conn, SoupWebsocketDataType type,
					   GBytes *message, gpointer userdata);

void on_incoming_stream(GstElement *webrtc, GstPad *pad, gpointer userdata);

void on_offer_created(GstPromise *promise, gpointer user_data);

void stopFeed(GstElement *pipeline, void *user_data);

struct GSTContext
{
public:
	GstElement *element;
	GstElement *appsrc;
	guint sourceid;
	GstClockTime timestamp;
};

class GStreamerWebRTCSink::Detail
{
public:
	Detail(GStreamerWebRTCSinkProps &_props, std::function<void()> _step) : mProps(_props), mStep(_step), ctx(nullptr)
	{
	}

	~Detail()
	{
	}

	void setMetadata(framemetadata_sp &metadata)
	{
		self = this;
		frameType = metadata->getFrameType();

		auto inputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
		width = inputRawMetadata->getWidth();
		height = inputRawMetadata->getHeight();

	}

	void setProps(GStreamerWebRTCSinkProps &props)
	{
		mProps = props;
	}

	GStreamerWebRTCSinkProps getProps()
	{
		return mProps;
	}

	gboolean readData_actual(GSTContext *ctx)
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
				printf("push buffer returned %d \n", gstret);
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

	gboolean cleanup_and_quit_loop(const gchar *msg, enum AppState state)
	{
		if (msg)
			g_printerr("%s\n", msg);
		if (state > 0)
			app_state = state;

		if (ws_conn)
		{
			if (soup_websocket_connection_get_state(ws_conn) ==
				SOUP_WEBSOCKET_STATE_OPEN)
				/* This will call us again */
				soup_websocket_connection_close(ws_conn, 1000, "");
			else
				g_object_unref(ws_conn);
		}

		/* To allow usage as a GSourceFunc */
		return G_SOURCE_REMOVE;
	}

	gchar *get_string_from_json_object(JsonObject *object)
	{
		JsonNode *root;
		JsonGenerator *generator;
		gchar *text;

		/* Make it the root node */
		root = json_node_init_object(json_node_alloc(), object);
		generator = json_generator_new();
		json_generator_set_root(generator, root);
		text = json_generator_to_data(generator, NULL);

		/* Release everything */
		g_object_unref(generator);
		json_node_free(root);
		return text;
	}
	void handle_media_stream(GstPad *pad, GstElement *pipe, const char *convert_name,
							 const char *sink_name)
	{
		GstPad *qpad;
		GstElement *q, *conv, *resample, *sink;
		GstPadLinkReturn ret;

		g_print("Trying to handle stream with %s ! %s", convert_name, sink_name);

		q = gst_element_factory_make("queue", NULL);
		g_assert_nonnull(q);
		conv = gst_element_factory_make(convert_name, NULL);
		g_assert_nonnull(conv);
		sink = gst_element_factory_make(sink_name, NULL);
		g_assert_nonnull(sink);

		if (g_strcmp0(convert_name, "audioconvert") == 0)
		{
			/* Might also need to resample, so add it just in case.
     * Will be a no-op if it's not required. */
			resample = gst_element_factory_make("audioresample", NULL);
			g_assert_nonnull(resample);
			gst_bin_add_many(GST_BIN(pipe), q, conv, resample, sink, NULL);
			gst_element_sync_state_with_parent(q);
			gst_element_sync_state_with_parent(conv);
			gst_element_sync_state_with_parent(resample);
			gst_element_sync_state_with_parent(sink);
			gst_element_link_many(q, conv, resample, sink, NULL);
		}
		else
		{
			gst_bin_add_many(GST_BIN(pipe), q, conv, sink, NULL);
			gst_element_sync_state_with_parent(q);
			gst_element_sync_state_with_parent(conv);
			gst_element_sync_state_with_parent(sink);
			gst_element_link_many(q, conv, sink, NULL);
		}

		qpad = gst_element_get_static_pad(q, "sink");

		ret = gst_pad_link(pad, qpad);
		g_assert_cmphex(ret, ==, GST_PAD_LINK_OK);
	}

	void on_incoming_decodebin_stream_helper(GstElement *decodebin, GstPad *pad,
											 GstElement *pipe)
	{
		GstCaps *caps;
		const gchar *name;

		if (!gst_pad_has_current_caps(pad))
		{
			g_printerr("Pad '%s' has no caps, can't do anything, ignoring\n",
					   GST_PAD_NAME(pad));
			return;
		}

		caps = gst_pad_get_current_caps(pad);
		name = gst_structure_get_name(gst_caps_get_structure(caps, 0));

		if (g_str_has_prefix(name, "video"))
		{
			handle_media_stream(pad, pipe, "videoconvert", "autovideosink");
		}
		else if (g_str_has_prefix(name, "audio"))
		{
			handle_media_stream(pad, pipe, "audioconvert", "autoaudiosink");
		}
		else
		{
			g_printerr("Unknown pad %s, ignoring", GST_PAD_NAME(pad));
		}
	}

	void on_incoming_stream_helper(GstElement *webrtc, GstPad *pad, GstElement *pipe)
	{
		GstElement *decodebin;

		if (GST_PAD_DIRECTION(pad) != GST_PAD_SRC)
			return;

		decodebin = gst_element_factory_make("decodebin", NULL);

		g_signal_connect(decodebin, "pad-added",
						 G_CALLBACK(on_incoming_decodebin_stream), this);
		gst_bin_add(GST_BIN(pipe), decodebin);
		gst_element_sync_state_with_parent(decodebin);
		gst_element_link(webrtc, decodebin);
	}

	void send_ice_candidate_message_helper(GstElement *webrtc G_GNUC_UNUSED, guint mlineindex,
										   gchar *candidate)
	{
		gchar *text;
		JsonObject *ice, *msg;

		if (app_state < PEER_CALL_NEGOTIATING)
		{
			cleanup_and_quit_loop("Can't send ICE, not in call", APP_STATE_ERROR);
			return;
		}

		ice = json_object_new();
		json_object_set_string_member(ice, "candidate", candidate);
		json_object_set_int_member(ice, "sdpMLineIndex", mlineindex);
		msg = json_object_new();
		json_object_set_object_member(msg, "ice", ice);
		text = get_string_from_json_object(msg);
		json_object_unref(msg);

		soup_websocket_connection_send_text(ws_conn, text);
		g_free(text);
	}

	void send_sdp_offer(GstWebRTCSessionDescription *offer)
	{
		gchar *text;
		JsonObject *msg, *sdp;

		if (app_state < PEER_CALL_NEGOTIATING)
		{
			cleanup_and_quit_loop("Can't send offer, not in call", APP_STATE_ERROR);
			return;
		}

		text = gst_sdp_message_as_text(offer->sdp);
		g_print("Sending offer:\n%s\n", text);

		sdp = json_object_new();
		json_object_set_string_member(sdp, "type", "offer");
		json_object_set_string_member(sdp, "sdp", text);
		g_free(text);

		msg = json_object_new();
		json_object_set_object_member(msg, "sdp", sdp);
		text = get_string_from_json_object(msg);
		json_object_unref(msg);

		soup_websocket_connection_send_text(ws_conn, text);
		g_free(text);
	}

	void on_offer_created_helper(GstPromise *promise)
	{
		GstWebRTCSessionDescription *offer = NULL;
		const GstStructure *reply;

		g_assert_cmphex(app_state, ==, PEER_CALL_NEGOTIATING);

		g_assert_cmphex(gst_promise_wait(promise), ==, GST_PROMISE_RESULT_REPLIED);
		reply = gst_promise_get_reply(promise);
		gst_structure_get(reply, "offer",
						  GST_TYPE_WEBRTC_SESSION_DESCRIPTION, &offer, NULL);
		gst_promise_unref(promise);

		promise = gst_promise_new();
		g_signal_emit_by_name(webrtc1, "set-local-description", offer, promise);
		gst_promise_interrupt(promise);
		gst_promise_unref(promise);

		/* Send offer to peer */
		send_sdp_offer(offer);
		gst_webrtc_session_description_free(offer);
	}

	void on_negotiation_needed_helper(GstElement *element, gpointer user_data)
	{
		GstPromise *promise;

		app_state = PEER_CALL_NEGOTIATING;
		promise = gst_promise_new_with_change_func(on_offer_created, user_data, NULL);
		g_signal_emit_by_name(webrtc1, "create-offer", NULL, promise);
	}

	void
	uridecodebin_element_added_helper(GstBin *bin,
									  GstElement *element)
	{
		if (g_strcmp0(gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(gst_element_get_factory(element))), "rtspsrc") == 0)
		{
			g_print("Added latency 100 ms to rtspsrc\n");
			g_object_set(G_OBJECT(element), "latency", 0,
						 "drop-on-latency", TRUE, NULL);
		}
	}

	gboolean
	start_pipeline(void)
	{
		GstStateChangeReturn ret;
		GError *error = NULL;

		LOG_INFO << "Configured raw frames pipeline.";
		pipe = gst_parse_launch("appsrc is-live=TRUE block=TRUE name=gstsrc min-latency=0 ! videoconvert ! video/x-raw, format=I420 ! x264enc tune=zerolatency speed-preset=superfast ! rtph264pay ! queue ! application/x-rtp,media=video,encoding-name=H264,payload=96 ! webrtcbin name=sendrecv", &error);

		if (error)
		{
			g_printerr("Failed to parse launch: %s\n", error->message);
			g_error_free(error);
			goto err;
		}

		ctx = g_new0(GSTContext, 1);

		// ctx->appsrc = gst_bin_get_by_name_recurse_up(GST_BIN(ctx->element),
		// 											 "gstsrc");
		ctx->appsrc = gst_bin_get_by_name(GST_BIN(pipe),
										  "gstsrc");

		gst_util_set_object_arg(G_OBJECT(ctx->appsrc), "format", "time");
		g_object_set(G_OBJECT(ctx->appsrc), "format", GST_FORMAT_TIME, NULL);


		g_object_set(G_OBJECT(ctx->appsrc), "caps",
						gst_caps_new_simple("video/x-raw",
											"format", G_TYPE_STRING, "RGB",
											"width", G_TYPE_INT, (gint)mProps.width,
											"height", G_TYPE_INT, (gint)mProps.height,
											"framerate", GST_TYPE_FRACTION, mProps.fps, 1, NULL),
						NULL);

		g_signal_connect(ctx->appsrc, "need-data", G_CALLBACK(startFeed), self);
		g_signal_connect(ctx->appsrc, "enough-data", G_CALLBACK(stopFeed), self);

		GstCaps *deco_caps;
		deco_caps = gst_caps_from_string(KMS_AGNOSTIC_NO_RTP_CAPS);

		gst_caps_unref(deco_caps);

		webrtc1 = gst_bin_get_by_name(GST_BIN(pipe), "sendrecv");
		g_assert_nonnull(webrtc1);

		/* This is the gstwebrtc entry point where we create the offer and so on. It
   * will be called when the pipeline goes to PLAYING. */
		g_signal_connect(webrtc1, "on-negotiation-needed",
						 G_CALLBACK(on_negotiation_needed), this);
		/* We need to transmit this ICE candidate to the browser via the websockets
   * signalling server. Incoming ice candidates from the browser need to be
   * added by us too, see on_server_message() */
		g_signal_connect(webrtc1, "on-ice-candidate",
						 G_CALLBACK(send_ice_candidate_message), this);
		/* Incoming streams will be exposed via this signal */
		g_signal_connect(webrtc1, "pad-added", G_CALLBACK(on_incoming_stream),
						 this);
		/* Lifetime is the same as the pipeline itself */
		gst_object_unref(webrtc1);

		g_print("Starting pipeline\n");
		ret = gst_element_set_state(GST_ELEMENT(pipe), GST_STATE_PLAYING);
		if (ret == GST_STATE_CHANGE_FAILURE)
			goto err;

		g_print("Started pipeline\n");
		return TRUE;

	err:
		if (pipe)
			g_clear_object(&pipe);
		if (webrtc1)
			webrtc1 = NULL;
		return FALSE;
	}

	gboolean
	setup_call(void)
	{
		gchar *msg;

		if (soup_websocket_connection_get_state(ws_conn) !=
			SOUP_WEBSOCKET_STATE_OPEN)
			return FALSE;

		if (!mProps.peerId.c_str())
			return FALSE;

		g_print("Setting up signalling server call with %s\n", mProps.peerId.c_str());
		app_state = PEER_CONNECTING;
		msg = g_strdup_printf("SESSION %s", mProps.peerId.c_str());
		// soup_websocket_connection_send_text(ws_conn, msg);
		g_free(msg);
		return TRUE;
	}

	gboolean
	register_with_server(void)
	{
		gchar *hello;
		gint32 our_id;

		if (soup_websocket_connection_get_state(ws_conn) !=
			SOUP_WEBSOCKET_STATE_OPEN)
			return FALSE;

		our_id = g_random_int_range(666, 666);
		g_print("Registering id %i with server\n", our_id);
		app_state = SERVER_REGISTERING;

		/* Register with the server with a random integer id. Reply will be received
   * by on_server_message() */
		hello = g_strdup_printf("HELLO %i", our_id);
		soup_websocket_connection_send_text(ws_conn, hello);
		g_free(hello);

		return TRUE;
	}

	void
	on_server_closed_helper(SoupWebsocketConnection *conn G_GNUC_UNUSED)
	{
		app_state = SERVER_CLOSED;
		cleanup_and_quit_loop("Server connection closed", APP_STATE_UNKNOWN);
		connect_to_websocket_server_async();
		return;
	}

	void
	on_server_message_helper(SoupWebsocketConnection *conn, SoupWebsocketDataType type,
							 GBytes *message)
	{
		gsize size;
		gchar *text, *data;

		switch (type)
		{
		case SOUP_WEBSOCKET_DATA_BINARY:
			g_printerr("Received unknown binary message, ignoring\n");
			g_bytes_unref(message);
			return;
		case SOUP_WEBSOCKET_DATA_TEXT:
			data = static_cast<gchar *>(g_bytes_unref_to_data(message, &size));
			/* Convert to NULL-terminated string */
			text = g_strndup(data, size);
			g_free(data);
			break;
		default:
			g_assert_not_reached();
		}

		/* Server has accepted our registration, we are ready to send commands */
		if (g_strcmp0(text, "HELLO") == 0)
		{
			if (app_state != SERVER_REGISTERING)
			{
				cleanup_and_quit_loop("ERROR: Received HELLO when not registering",
									  APP_STATE_ERROR);
				goto out;
			}
			app_state = SERVER_REGISTERED;
			g_print("Registered with server\n");
			/* Ask signalling server to connect us with a specific peer */
			if (!setup_call())
			{
				cleanup_and_quit_loop("ERROR: Failed to setup call", PEER_CALL_ERROR);
				goto out;
			}
			/* Call has been setup by the server, now we can start negotiation */
		}
		else if (g_strcmp0(text, "SESSION_OK") == 0)
		{
			if (app_state != PEER_CONNECTING)
			{
				cleanup_and_quit_loop("ERROR: Received SESSION_OK when not calling",
									  PEER_CONNECTION_ERROR);
				goto out;
			}

			app_state = PEER_CONNECTED;
			/* Start negotiation (exchange SDP and ICE candidates) */
			LOG_INFO << "Going to start pipeline";
			if (!start_pipeline())
				cleanup_and_quit_loop("ERROR: failed to start pipeline",
									  PEER_CALL_ERROR);
			/* Handle errors */
		}
		else if (g_str_has_prefix(text, "ERROR"))
		{
			switch (app_state)
			{
			case SERVER_CONNECTING:
				app_state = SERVER_CONNECTION_ERROR;
				break;
			case SERVER_REGISTERING:
				app_state = SERVER_REGISTRATION_ERROR;
				break;
			case PEER_CONNECTING:
				app_state = PEER_CONNECTION_ERROR;
				break;
			case PEER_CONNECTED:
			case PEER_CALL_NEGOTIATING:
				app_state = PEER_CALL_ERROR;
			default:
				app_state = APP_STATE_ERROR;
			}
			cleanup_and_quit_loop(text, APP_STATE_UNKNOWN);
			/* Look for JSON messages containing SDP and ICE candidates */
		}
		else
		{
			JsonNode *root;
			JsonObject *object, *child;
			JsonParser *parser = json_parser_new();
			if (!json_parser_load_from_data(parser, text, -1, NULL))
			{
				g_printerr("Unknown message '%s', ignoring", text);
				g_object_unref(parser);
				goto out;
			}

			root = json_parser_get_root(parser);
			if (!JSON_NODE_HOLDS_OBJECT(root))
			{
				g_printerr("Unknown json message '%s', ignoring", text);
				g_object_unref(parser);
				goto out;
			}

			object = json_node_get_object(root);
			/* Check type of JSON message */
			if (json_object_has_member(object, "sdp"))
			{
				int ret;
				GstSDPMessage *sdp;
				const gchar *text, *sdptype;
				GstWebRTCSessionDescription *answer;

				g_assert_cmphex(app_state, ==, PEER_CALL_NEGOTIATING);

				child = json_object_get_object_member(object, "sdp");

				if (!json_object_has_member(child, "type"))
				{
					cleanup_and_quit_loop("ERROR: received SDP without 'type'",
										  PEER_CALL_ERROR);
					goto out;
				}

				sdptype = json_object_get_string_member(child, "type");
				/* In this example, we always create the offer and receive one answer.
       * See tests/examples/webrtcbidirectional.c in gst-plugins-bad for how to
       * handle offers from peers and reply with answers using webrtcbin. */
				g_assert_cmpstr(sdptype, ==, "answer");

				text = json_object_get_string_member(child, "sdp");

				g_print("Received answer:\n%s\n", text);

				ret = gst_sdp_message_new(&sdp);
				g_assert_cmphex(ret, ==, GST_SDP_OK);

				ret = gst_sdp_message_parse_buffer((guint8 *)text, strlen(text), sdp);
				g_assert_cmphex(ret, ==, GST_SDP_OK);

				answer = gst_webrtc_session_description_new(GST_WEBRTC_SDP_TYPE_ANSWER,
															sdp);
				g_assert_nonnull(answer);

				/* Set remote description on our pipeline */
				{
					GstPromise *promise = gst_promise_new();
					g_signal_emit_by_name(webrtc1, "set-remote-description", answer,
										  promise);
					gst_promise_interrupt(promise);
					gst_promise_unref(promise);
				}

				app_state = PEER_CALL_STARTED;
			}
			else if (json_object_has_member(object, "ice"))
			{
				const gchar *candidate;
				gint sdpmlineindex;

				child = json_object_get_object_member(object, "ice");
				candidate = json_object_get_string_member(child, "candidate");
				sdpmlineindex = json_object_get_int_member(child, "sdpMLineIndex");

				/* Add ice candidate sent by remote peer */
				g_signal_emit_by_name(webrtc1, "add-ice-candidate", sdpmlineindex,
									  candidate);
			}
			else
			{
				g_printerr("Ignoring unknown JSON message:\n%s\n", text);
			}
			g_object_unref(parser);
		}

	out:
		g_free(text);
	}

	void
	on_server_connected_helper(SoupSession *session, GAsyncResult *res,
							   SoupMessage *msg)
	{
		GError *error = NULL;

		ws_conn = soup_session_websocket_connect_finish(session, res, &error);
		if (error)
		{
			cleanup_and_quit_loop(error->message, SERVER_CONNECTION_ERROR);
			g_error_free(error);
			return;
		}

		g_assert_nonnull(ws_conn);

		app_state = SERVER_CONNECTED;
		g_print("Connected to signalling server\n");

		g_signal_connect(ws_conn, "closed", G_CALLBACK(on_server_closed), this);
		g_signal_connect(ws_conn, "message", G_CALLBACK(on_server_message), this);

		/* Register with the server so it knows about us and can accept commands */
		register_with_server();
	}

	void
	connect_to_websocket_server_async(void)
	{
		SoupLogger *logger;
		SoupSession *session;
		const char *https_aliases[] = {"wss", NULL};

		session = soup_session_new_with_options(SOUP_SESSION_SSL_STRICT, !disable_ssl,
												SOUP_SESSION_SSL_USE_SYSTEM_CA_FILE, TRUE,
												//SOUP_SESSION_SSL_CA_FILE, "/etc/ssl/certs/ca-bundle.crt",
												SOUP_SESSION_HTTPS_ALIASES, https_aliases, NULL);

		logger = soup_logger_new(SOUP_LOGGER_LOG_BODY, -1);
		soup_session_add_feature(session, SOUP_SESSION_FEATURE(logger));
		g_object_unref(logger);

		message = soup_message_new(SOUP_METHOD_GET, server_url);

		g_print("Connecting to server...\n");

		/* Once connected, we will register */
		soup_session_websocket_connect_async(session, message, NULL, NULL, NULL,
											 (GAsyncReadyCallback)on_server_connected, this);
		app_state = SERVER_CONNECTING;
	}

	gboolean
	check_plugins(void)
	{
		int i;
		gboolean ret;
		GstPlugin *plugin;
		GstRegistry *registry;
		const gchar *needed[] = {"opus", "vpx", "nice", "webrtc", "dtls", "srtp",
								 "rtpmanager", "videotestsrc", "audiotestsrc", NULL};

		registry = gst_registry_get();
		ret = TRUE;
		for (i = 0; i < g_strv_length((gchar **)needed); i++)
		{
			plugin = gst_registry_find_plugin(registry, needed[i]);
			if (!plugin)
			{
				g_print("Required gstreamer plugin '%s' not found\n", needed[i]);
				ret = FALSE;
				continue;
			}
			gst_object_unref(plugin);
		}
		return ret;
	}

	bool gstreamerMainThread(GMainLoop *loop)
	{

		gst_init(0, NULL);

		loop = g_main_loop_new(NULL, FALSE);

		{
			GstUri *uri = gst_uri_from_string(server_url);
			if (g_strcmp0("localhost", gst_uri_get_host(uri)) == 0 ||
				g_strcmp0("127.0.0.1", gst_uri_get_host(uri)) == 0)
				disable_ssl = TRUE;
			gst_uri_unref(uri);
		}

		connect_to_websocket_server_async();
		g_print("Reached after server close.");
		g_main_loop_run(loop);
		g_main_loop_unref(loop);

		return true;
	}

public:
	GStreamerWebRTCSinkProps mProps;
	GstBuffer *buffer;
	GSTContext *ctx;
	GMainLoop *loop;
	SoupMessage *message;
	GstElement *pipe, *webrtc1;
	SoupWebsocketConnection *ws_conn = NULL;
	enum AppState app_state = APP_STATE_UNKNOWN;
	const gchar *server_url = "ws://127.0.0.1:8083/signalling";
	gboolean disable_ssl = FALSE;
	frame_sp frame;
	int width = 0;
	int height;
	char desc[10000];
	FrameMetadata::FrameType frameType;
	std::function<void()> mStep;
	Detail *self;
	std::thread t1;
};

void startFeed(GstElement *pipeline, guint size, void *user_data)
{
	auto detail = static_cast<GStreamerWebRTCSink::Detail *>(user_data);
	if (detail->ctx->sourceid == 0)
	{
		detail->ctx->sourceid = g_idle_add((GSourceFunc)readData, user_data);
	}
	return;
}

void on_incoming_decodebin_stream(GstElement *decodebin, GstPad *pad,
								  void *userdata)
{
	auto detail = static_cast<GStreamerWebRTCSink::Detail *>(userdata);
	detail->on_incoming_decodebin_stream_helper(decodebin, pad, detail->pipe);
	return;
}

void on_offer_created(GstPromise *promise, gpointer userdata)
{
	auto detail = static_cast<GStreamerWebRTCSink::Detail *>(userdata);
	detail->on_offer_created_helper(promise);
	return;
}

void on_negotiation_needed(GstElement *element, gpointer userdata)
{
	auto detail = static_cast<GStreamerWebRTCSink::Detail *>(userdata);
	detail->on_negotiation_needed_helper(element, userdata);
	return;
}

void send_ice_candidate_message(GstElement *webrtc G_GNUC_UNUSED, guint mlineindex,
								gchar *candidate, gpointer userdata)
{
	auto detail = static_cast<GStreamerWebRTCSink::Detail *>(userdata);
	detail->send_ice_candidate_message_helper(webrtc, mlineindex, candidate);
	return;
}

void on_incoming_stream(GstElement *webrtc, GstPad *pad, gpointer userdata)
{
	auto detail = static_cast<GStreamerWebRTCSink::Detail *>(userdata);
	detail->on_incoming_stream_helper(webrtc, pad, detail->pipe);
	return;
}

void uridecodebin_element_added(GstBin *bin,
								GstElement *element, gpointer userdata)
{
	auto detail = static_cast<GStreamerWebRTCSink::Detail *>(userdata);
	detail->uridecodebin_element_added_helper(bin, element);
	return;
}

void on_server_closed(SoupWebsocketConnection *conn G_GNUC_UNUSED,
					  gpointer userdata)
{
	auto detail = static_cast<GStreamerWebRTCSink::Detail *>(userdata);
	detail->on_server_closed_helper(conn);
	return;
}

void on_server_connected(SoupSession *session, GAsyncResult *res,
						 gpointer userdata)
{
	auto detail = static_cast<GStreamerWebRTCSink::Detail *>(userdata);
	detail->on_server_connected_helper(session, res, detail->message);
	return;
}

void on_server_message(SoupWebsocketConnection *conn, SoupWebsocketDataType type,
					   GBytes *message, gpointer userdata)
{
	auto detail = static_cast<GStreamerWebRTCSink::Detail *>(userdata);
	detail->on_server_message_helper(conn, type, message);
	return;
}

void readData(void *user_data)
{
	auto detail = static_cast<GStreamerWebRTCSink::Detail *>(user_data);
	detail->mStep();
	return;
}

void stopFeed(GstElement *pipeline, void *user_data)
{
	auto detail = static_cast<GStreamerWebRTCSink::Detail *>(user_data);
	if (detail->ctx->sourceid != 0)
	{
		g_source_remove(detail->ctx->sourceid);
		detail->ctx->sourceid = 0;
	}
	return;
}

GStreamerWebRTCSink::GStreamerWebRTCSink(GStreamerWebRTCSinkProps props) : Module(SINK, "GStreamerWebRTCSink", props)
{
	mDetail.reset(new Detail(props, [&]() -> void
							 { Module::step(); }));
}

GStreamerWebRTCSink::~GStreamerWebRTCSink() {}

bool GStreamerWebRTCSink::init()
{
	if (!Module::init())
	{
		return false;
	}

	framemetadata_sp metadata = getFirstInputMetadata();
	mDetail->setMetadata(metadata);

	mDetail->t1 = std::thread(&GStreamerWebRTCSink::Detail::gstreamerMainThread, mDetail.get(), mDetail->loop);

	return true;
}

bool GStreamerWebRTCSink::term()
{

	g_main_loop_quit(mDetail->loop);
	mDetail->t1.join();
	gst_deinit();
	return Module::term();
}

bool GStreamerWebRTCSink::validateInputPins()
{
	if (getNumberOfInputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstInputMetadata();

	FrameMetadata::FrameType frameType = metadata->getFrameType();
	LOG_INFO << "FrameType is " << frameType;
	if (frameType != FrameMetadata::RAW_IMAGE)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be RAW. Actual<" << frameType << ">";
		return false;
	}
	return true;
}

bool GStreamerWebRTCSink::process(frame_container &frames)
{
	mDetail->frame = frames.cbegin()->second;
	mDetail->readData_actual(mDetail->ctx);
	return true;
}

bool GStreamerWebRTCSink::processSOS(frame_sp &frame)
{
	return true;
}

bool GStreamerWebRTCSink::shouldTriggerSOS()
{
	return mDetail->width == 0;
}

bool GStreamerWebRTCSink::processEOS(string &pinId)
{
	return false;
}

GStreamerWebRTCSinkProps GStreamerWebRTCSink::getProps()
{
	auto mProps = mDetail->getProps();
	fillProps(mProps);

	return mProps;
}

void GStreamerWebRTCSink::setProps(GStreamerWebRTCSinkProps &mProps)
{
	Module::addPropsToQueue(mProps);
}

bool GStreamerWebRTCSink::handlePropsChange(frame_sp &frame)
{
	GStreamerWebRTCSinkProps mProps;
	bool ret = Module::handlePropsChange(frame, mProps);
	mDetail->setProps(mProps);

	sendEOS();

	return ret;
}
