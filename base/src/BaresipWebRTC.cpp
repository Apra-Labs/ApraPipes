#include <stdafx.h>
#include "BaresipWebRTC.h"
#include <re.h>
#include <baresip.h>
#include <re_dbg.h>
#include "BaresipDemo.h"
#include <stdio.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <rem.h>
#include <cstdlib>

#ifdef HAVE_GETOPT
#include <getopt.h>
#endif
#define DEBUG_LEVEL 6
enum { ASYNC_WORKERS = 4 };

static const char *modpath = "/usr/local/lib/baresip/modules";
static const char *server_cert = "/etc/demo.pem";
static const char *www_path = "webrtc/www";

//video source vidsrc
static struct vidsrc *vidsrc;




//video source state
struct vidsrc_st {
   int filedesc;
   thrd_t thread;
   bool isRunning;
   struct vidsz vidsize;
   u_int32_t pixfmt;
   vidsrc_frame_h *frameh;
   unsigned char* buffer;
   void *arg;
};

struct vidsrc_st *st_point;
bool startWrite = false;

static void destructor(void *arg)
{
	   struct vidsrc_st *st = (vidsrc_st *)arg;
   debug("vidpipe: stopping video source..\n");
   if (st->isRunning) {
       st->isRunning = false;
	   free(st->buffer);
       startWrite = false;
	   //thrd_join(st->thread, NULL);
	   //hello;
   }
}


static const char *modv[] = {
	"ice",
	"dtls_srtp",

	/* audio */
	"opus",
	"g711",
	"ausine",

	/* video */
	"vp8",
	"avcodec",
	"vp9",
    "v4l2",
	//"avformat",
	"fakevideo"
};

static const char *ice_server = NULL;

static const char *modconfig =
	"opus_bitrate       96000\n"
	"opus_stereo        yes\n"
	"opus_sprop_stereo  yes\n"
	"\n"
	"avformat_pass_through  no\n"
	;


static void signal_handler(int signum)
{
	(void)signum;

	re_fprintf(stderr, "terminated on signal %d\n", signum);

	re_cancel();
}


static void usage(void)
{
	re_fprintf(stderr,
		   "Usage: baresip-webrtc [options]\n"
		   "\n"
		   "options:\n"
                   "\t-h               Help\n"
		   "\t-v               Verbose debug\n"
		   "\n"
		   "http:\n"
		   "\t-c <cert>        HTTP server certificate (%s)\n"
		   "\t-w <root>        HTTP server document root (%s)\n"
		   "\n"
		   "ice:\n"
		   "\t-i <server>      ICE server (%s)\n"
		   "\t-u <username>    ICE username\n"
		   "\t-p <password>    ICE password\n"
		   "\n",
		   server_cert,
		   www_path,
		   ice_server);
}

void vidframe_init_buf(struct vidframe *vf, enum vidfmt fmt,
		       const struct vidsz *sz, uint8_t *buf)
{
	unsigned w, h;

	if (!vf || !sz || !buf)
		return;

	w = (sz->w + 1) >> 1;
	h = (sz->h + 1) >> 1;

	unsigned w2 = (sz->w + 1) >> 1;

	memset(vf->linesize, 0, sizeof(vf->linesize));
	memset(vf->data, 0, sizeof(vf->data));

	switch (fmt) {

	case VID_FMT_YUV420P:
		vf->linesize[0] = sz->w;
		vf->linesize[1] = w;
		vf->linesize[2] = w;

		vf->data[0] = buf;
		vf->data[1] = vf->data[0] + vf->linesize[0] * sz->h;
		vf->data[2] = vf->data[1] + vf->linesize[1] * h;
		break;

	case VID_FMT_YUYV422:
	case VID_FMT_UYVY422:
		vf->linesize[0] = sz->w * 2;
		vf->data[0] = buf;
		break;

	case VID_FMT_RGB32:
	case VID_FMT_ARGB:
		vf->linesize[0] = sz->w * 4;
		vf->data[0] = buf;
		break;

	case VID_FMT_RGB565:
		vf->linesize[0] = sz->w * 2;
		vf->data[0] = buf;
		break;

	case VID_FMT_NV12:
	case VID_FMT_NV21:
		vf->linesize[0] = sz->w;
		vf->linesize[1] = w*2;

		vf->data[0] = buf;
		vf->data[1] = vf->data[0] + vf->linesize[0] * sz->h;
		break;

	case VID_FMT_YUV444P:
		vf->linesize[0] = sz->w;
		vf->linesize[1] = sz->w;
		vf->linesize[2] = sz->w;

		vf->data[0] = buf;
		vf->data[1] = vf->data[0] + vf->linesize[0] * sz->h;
		vf->data[2] = vf->data[1] + vf->linesize[1] * sz->h;
		break;

	case VID_FMT_YUV422P:
		vf->linesize[0] = sz->w;
		vf->linesize[1] = w2;
		vf->linesize[2] = w2;

		vf->data[0] = buf;
		vf->data[1] = vf->data[0] + vf->linesize[0] * sz->h;
		vf->data[2] = vf->data[1] + vf->linesize[1] * sz->h;
		break;

	default:
		//(void)re_printf("vidframe: no fmt %s\n", vidfmt_name(fmt));
		return;
	}

	vf->size = *sz;
	vf->fmt = fmt;
}


BaresipWebRTC::BaresipWebRTC(BaresipWebRTCProps _props)
{
}

BaresipWebRTC::~BaresipWebRTC() {}

//reading frame


static int read_frame(struct vidsrc_st *st)
{
   // Creating buffer for contents
   size_t read_size = st->vidsize.w*st->vidsize.h*1.5;
   unsigned char* Y = st->buffer;
   for(int height = 0; height < st->vidsize.h; height++)
   {
       //Loop of rows
       uint8_t shade = (uint8_t)(height%256);
       memset(Y, shade, st->vidsize.w);
       Y+=st->vidsize.w;
   }


   struct timeval ts;
   uint64_t timestamp;
   struct vidframe frame;
   st->pixfmt = 0;
   vidframe_init_buf(&frame, VID_FMT_YUV420P, &st->vidsize, st->buffer);


   gettimeofday(&ts,NULL);
   timestamp = 1000000U * ts.tv_sec + ts.tv_usec;
   timestamp = timestamp * VIDEO_TIMEBASE / 1000000U;


   st->frameh(&frame, timestamp, st->arg);

	return 0;
   //free(buffer);
}



static int read_thread(void *arg)
{
   struct vidsrc_st *st = (vidsrc_st*)arg;
   st_point = st;
   int err;
   startWrite = true;
   return 0;
}


static int pipe_alloc(struct vidsrc_st **stp, const struct vidsrc *vs,
		 struct vidsrc_prm *prm,
		 const struct vidsz *size, const char *fmt,
		 const char *dev, vidsrc_frame_h *frameh,
		 vidsrc_packet_h  *packeth,
		 vidsrc_error_h *errorh, void *arg)
{

	info("I am reaching pipe_alloc");
	struct vidsrc_st *st;
   int err;


   (void)prm;
   (void)fmt;
   (void)packeth;
   (void)errorh;


   if (!stp || !size || !frameh)
       return EINVAL;


   st = (vidsrc_st*)(mem_zalloc(sizeof(*st), destructor));
   if (!st)
       return ENOMEM;


//initialize size


   st->filedesc = -1;
   st->vidsize = *size;
   st->vidsize.w = 640;
   st->vidsize.h = 360;
   st->frameh = frameh;
   st->arg    = arg;
   st->pixfmt = VID_FMT_YUV420P;


   size_t read_size = st->vidsize.w*st->vidsize.h*1.5;
   st->buffer = (unsigned char*)malloc(read_size);
   memset(st->buffer,0, read_size);


   st->isRunning = true;
   err = thread_create_name(&st->thread, "vidpipe", read_thread, st);
   if (err) {
       st->isRunning = false;
       goto out;
   }


out:
   if (err)
       mem_deref(st);
   else
       *stp = st;


   return err;

}



bool BaresipWebRTC::init(int argc, char* argv[]) 
{
    //printf("Arguement 1 : %d\n", argc);

    //for (int i = 0; i < argc; i++) {
    //    printf("Argument 2's  %d: %s\n", i, argv[i]);
    //}
    struct config *config;
	const char *stun_user = NULL, *stun_pass = NULL;
	int err = 0;

    #ifdef HAVE_GETOPT
	    for (;;) {

		    const int c = getopt(argc, argv, "c:hl:i:u:tvu:p:w:");
		    if (0 > c)
			    break;

		    switch (c) {

		    case '?':
		    default:
			    err = EINVAL;
			    /*@fallthrough@*/

		    case 'c':
			    server_cert = optarg;
			    break;

		    case 'h':
			    usage();
			    return err;

		    case 'i':
			    if (0 == str_casecmp(optarg, "null"))
				    ice_server = NULL;
			    else
				    ice_server = optarg;
			    break;

		    case 'u':
			    stun_user = optarg;
			    break;

		    case 'p':
			    stun_pass = optarg;
			    break;

		    case 'v':
			    log_enable_debug(true);
			    break;

		    case 'w':
			    www_path = optarg;
			    break;
		    }
	    }

	    if (argc < 1 || (argc != (optind + 0))) {
		    usage();
		    return -2;
	    }
    #else
	    (void)argc;
	    (void)argv;
    #endif

	err = libre_init();
	if (err) {
		re_fprintf(stderr, "libre_init: %m\n", err);
		close();
	}

	re_thread_async_init(ASYNC_WORKERS);

	sys_coredump_set(true);

	err = conf_configure_buf((uint8_t *)modconfig, str_len(modconfig));
	if (err) {
		warning("main: configure failed: %m\n", err);
		close();
	}//libmp4 is a C library to handle MP4 files (ISO base media file format, see ISO/IEC 14496-12). It is mainly targeted to be used with videos produced by Parrot Drones (Bebop, Bebop2, Disco, etc.).

	config = conf_config();

	config->net.use_linklocal = false;

	/*
	 * Initialise the top-level baresip struct, must be
	 * done AFTER configuration is complete.
	 */
	err = baresip_init(conf_config());
	if (err) {
		warning("main: baresip init failed (%m)\n", err);
		close();
	}

	for (size_t i=0; i<RE_ARRAY_SIZE(modv); i++) {

		err = module_load(modpath, modv[i]);
		if (err) {
			re_fprintf(stderr,
				   "could not pre-load module"
				   " '%s' (%m)\n", modv[i], err);
		}
	}

	str_ncpy(config->audio.src_mod, "ausine",
		 sizeof(config->audio.src_mod));
	str_ncpy(config->audio.src_dev,
		 "440",
		 sizeof(config->audio.src_dev));
	err = vidsrc_register(&myVidsrc, baresip_vidsrcl(),
			       "vidpipe", pipe_alloc, NULL);
	str_ncpy(config->video.src_mod, "vidpipe",
		 sizeof(config->video.src_mod));
	//str_ncpy(config->video.src_dev, "/dev/video0",
	//	 sizeof(config->video.src_dev));

	config->audio.level = true;

	config->video.bitrate = 2000000;
	config->video.fps = 30.0;
	config->video.fullscreen = false;
	config->video.width  = 640;
	config->video.height = 360;

	/* override default config */
	config->avt.rtcp_mux = true;
	config->avt.rtp_stats = true;

	err = demo_init(server_cert, www_path,
			ice_server, stun_user, stun_pass);
	if (err) {
		re_fprintf(stderr, "failed to init demo: %m\n", err);
		close();
	}
    return true;
}

bool BaresipWebRTC::term()
{
    close();
    return true;
}

bool BaresipWebRTC::processSOS()
{
    pthread_t thread_id;
    pthread_create(&thread_id, NULL,(void*(*)(void *))re_main,(void *)signal_handler);
    pthread_detach(thread_id);
    return true;
}

bool BaresipWebRTC::process(void *frame_data)
{
	if(startWrite)
	{
		// Creating buffer for contents
   		size_t read_size = st_point->vidsize.w*st_point->vidsize.h*1.5;

		//memcpy(Y,frame_data,read_size);
		st_point->buffer = static_cast<unsigned char*>(frame_data);

		struct timeval ts;
		uint64_t timestamp;
		struct vidframe frame;
		st_point->pixfmt = 0;
		vidframe_init_buf(&frame, VID_FMT_YUV420P, &st_point->vidsize, st_point->buffer);


		gettimeofday(&ts,NULL);
		timestamp = 1000000U * ts.tv_sec + ts.tv_usec;
		timestamp = timestamp * VIDEO_TIMEBASE / 1000000U;


		st_point->frameh(&frame, timestamp, st_point->arg);
	}

    return true;
}

void BaresipWebRTC::close()
{
    demo_close();

	/* note: must be done before mod_close() */
	module_app_unload();

	conf_close();

	baresip_close();

	/* NOTE: modules must be unloaded after all application
	 *       activity has stopped.
	 */
	debug("main: unloading modules..\n");
	mod_close();

	re_thread_async_close();

	tmr_debug();

	libre_close();

	/* Check for memory leaks */
	mem_debug();
}
