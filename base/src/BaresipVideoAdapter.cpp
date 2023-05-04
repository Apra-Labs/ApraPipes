#include <stdafx.h>
#include <stdio.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/shm.h>
#include <string.h>
#include <map>
#include <stdlib.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#ifdef HAVE_GETOPT
#include <getopt.h>
#endif
#include <re.h>
#include <baresip.h>
#include <boost/thread.hpp>
#define DEBUG_MODULE ""
#define DEBUG_LEVEL 0
#include <re_dbg.h>

#include "BaresipVideoAdapter.h"

enum
{
	ASYNC_WORKERS = 4
};

bool startWriting = false;

static void signal_handler(int sig)
{
	static bool term = false;

	if (term)
	{
		module_app_unload();
		mod_close();
		exit(0);
	}

	term = true;

	info("terminated by signal %d\n", sig);

	ua_stop_all(false);
}

struct vidsz
{
	unsigned w; /**< Width */
	unsigned h; /**< Height */
};

enum vidfmt
{
	VID_FMT_YUV420P = 0, /* planar YUV 4:2:0 12bpp */
	VID_FMT_YUYV422,	 /* packed YUV 4:2:2 16bpp */
	VID_FMT_UYVY422,	 /* packed YUV 4:2:2 16bpp */
	VID_FMT_RGB32,		 /* packed RGBA 8:8:8:8 32bpp (native endian) */
	VID_FMT_ARGB,		 /* packed RGBA 8:8:8:8 32bpp (big endian) */
	VID_FMT_RGB565,		 /* packed RGB 5:6:5 16bpp (native endian) */
	VID_FMT_NV12,		 /* planar YUV 4:2:0 12bpp UV interleaved */
	VID_FMT_NV21,		 /* planar YUV 4:2:0 12bpp VU interleaved */
	VID_FMT_YUV444P,	 /* planar YUV 4:4:4 24bpp */
	VID_FMT_YUV422P,	 /* planar YUV 4:2:2 16bpp */
	/* marker */
	VID_FMT_N
};

/** Video frame */
struct vidframe
{
	uint8_t *data[4];	  /**< Video planes */
	uint16_t linesize[4]; /**< Array of line-sizes */
	struct vidsz size;	  /**< Frame resolution */
	enum vidfmt fmt;	  /**< Video pixel format */
};

// video source state
struct vidsrc_st
{
	int filedesc;
	thrd_t thread;
	bool isRunning;
	struct vidsz vidsize;
	u_int32_t pixfmt;
	vidsrc_frame_h *frameh;
	unsigned char *buffer;
	void *arg;
};

struct vidsrc_st *st_ptr;

struct tmr tmr_quit;

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

	switch (fmt)
	{

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
		vf->linesize[1] = w * 2;

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

static void ua_exit_handler(void *arg)
{
	(void)arg;
	debug("ua exited -- stopping main runloop\n");

	/* The main run-loop can be stopped now */
	re_cancel();
}

static void tmr_quit_handler(void *arg)
{
	(void)arg;

	ua_stop_all(false);
}

static void usage(void)
{
	(void)re_fprintf(stderr,
					 "Usage: baresip [options]\n"
					 "options:\n"
					 "\t-4 Force IPv4 only\n"
#if HAVE_INET6
					 "\t-6 Force IPv6 only\n"
#endif
					 "\t-a <software> Specify SIP User-Agent string\n"
					 "\t-d Daemon\n"
					 "\t-e <commands> Execute commands (repeat)\n"
					 "\t-f <path> Config path\n"
					 "\t-m <module> Pre-load modules (repeat)\n"
					 "\t-p <path> Audio files\n"
					 "\t-h -? Help\n"
					 "\t-s Enable SIP trace\n"
					 "\t-t <sec> Quit after <sec> seconds\n"
					 "\t-n <net_if> Specify network interface\n"
					 "\t-u <parameters> Extra UA parameters\n"
					 "\t-v Verbose debug\n"
					 "\t-T Enable timestamps log\n"
					 "\t-c Disable colored log\n");
}

static void destructor(void *arg)
{
	struct vidsrc_st *st = (vidsrc_st *)arg;
	debug("vidpipe: stopping video source..\n");
	if (st->isRunning)
	{
		st->isRunning = false;
		thrd_join(st->thread, NULL);
		startWriting = false;
	}
}

BaresipVideoAdapter::BaresipVideoAdapter(BaresipVideoAdapterProps _props)
{
}

BaresipVideoAdapter::~BaresipVideoAdapter() {}

static int read_thread(void *arg)
{
	struct vidsrc_st *st = (vidsrc_st *)arg;
	st_ptr = st;
	int err;
	startWriting = true;

	return 0;
}

static int pipe_alloc(struct vidsrc_st **stp, const struct vidsrc *vs,
					  struct vidsrc_prm *prm,
					  const struct vidsz *size, const char *fmt,
					  const char *dev, vidsrc_frame_h *frameh,
					  vidsrc_packet_h *packeth,
					  vidsrc_error_h *errorh, void *arg)
{

	info("I am reaching pipe_alloc");
	//return 0;


	// from vidpipe code

	struct vidsrc_st *st;
	int err;

	(void)prm;
	(void)fmt;
	(void)packeth;
	(void)errorh;

	if (!stp || !size || !frameh)
		return EINVAL;

	st = (vidsrc_st *)mem_zalloc(sizeof(*st), destructor);
	if (!st)
		return ENOMEM;

	// initialize size

	st->filedesc = -1;
	st->vidsize = *size;
	st->vidsize.w = 640;
	st->vidsize.h = 360;
	st->frameh = frameh;
	st->arg = arg;
	st->pixfmt = VID_FMT_YUV420P;

	size_t read_size = st->vidsize.w * st->vidsize.h * 1.5;
	st->buffer = (unsigned char *)malloc(read_size);
	memset(st->buffer, 0, read_size);

	st->isRunning = true;
	err = thread_create_name(&st->thread, "vidpipe", start_reading, st);
	if (err)
	{
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

bool BaresipVideoAdapter::init()
{
	int argc = 0;
	char **argv = nullptr;
	int af = AF_UNSPEC, run_daemon = false;
	const char *ua_eprm = NULL;
	const char *software =
		"baresip v" BARESIP_VERSION " (" ARCH "/" OS ")";
	const char *execmdv[16];
	const char *net_interface = NULL;
	const char *audio_path = NULL;
	const char *modv[16];
	bool sip_trace = false;
	size_t execmdc = 0;
	size_t modc = 0;
	size_t i;
	uint32_t tmo = 0;
	int dbg_level = DBG_INFO;
	enum dbg_flags dbg_flags = DBG_ANSI;

	/*
	 * turn off buffering on stdout
	 */
	setbuf(stdout, NULL);

	(void)re_fprintf(stdout, "baresip v%s"
							 " Copyright (C) 2010 - 2022"
							 " Alfred E. Heggestad et al.\n",
					 BARESIP_VERSION);

	(void)sys_coredump_set(true);

	err = libre_init();
	if (err)
		term();

	tmr_init(&tmr_quit);

#ifdef HAVE_GETOPT
	for (;;)
	{
		const int c = getopt(argc, argv, "46a:de:f:p:hu:n:vst:m:Tc");
		if (0 > c)
			break;

		switch (c)
		{

		case '?':
		case 'h':
			usage();
			return -2;

		case 'a':
			software = optarg;
			break;

		case '4':
			af = AF_INET;
			break;

#if HAVE_INET6
		case '6':
			af = AF_INET6;
			break;
#endif

		case 'd':
			run_daemon = true;
			break;

		case 'e':
			if (execmdc >= RE_ARRAY_SIZE(execmdv))
			{
				warning("max %zu commands\n",
						RE_ARRAY_SIZE(execmdv));
				err = EINVAL;
				term();
			}
			execmdv[execmdc++] = optarg;
			break;

		case 'f':
			conf_path_set(optarg);
			break;

		case 'm':
			if (modc >= RE_ARRAY_SIZE(modv))
			{
				warning("max %zu modules\n",
						RE_ARRAY_SIZE(modv));
				err = EINVAL;
				term();
			}
			modv[modc++] = optarg;
			break;

		case 'p':
			audio_path = optarg;
			break;

		case 's':
			sip_trace = true;
			break;

		case 't':
			tmo = atoi(optarg);
			break;

		case 'n':
			net_interface = optarg;
			break;

		case 'u':
			ua_eprm = optarg;
			break;

		case 'v':
			log_enable_debug(true);
			dbg_level = DBG_DEBUG;
			break;

		case 'T':
			log_enable_timestamps(true);
			dbg_flags |= DBG_TIME;
			break;

		case 'c':
			log_enable_color(false);
			dbg_flags &= ~DBG_ANSI;
			break;

		default:
			break;
		}
	}
#else
	(void)argc;
	(void)argv;
#endif

	dbg_init(dbg_level, dbg_flags);
	err = conf_configure();
	if (err)
	{
		warning("main: configure failed: %m\n", err);
		term();
	}

	re_thread_async_init(ASYNC_WORKERS);

	/*
	 * Set the network interface before initializing the config
	 */
	if (net_interface)
	{
		struct config *theconf = conf_config();

		str_ncpy(theconf->net.ifname, net_interface,
				 sizeof(theconf->net.ifname));
	}

	/*
	 * Set prefer_ipv6 preferring the one given in -6 argument (if any)
	 */
	if (af != AF_UNSPEC)
		conf_config()->net.af = af;

	/*
	 * Initialise the top-level baresip struct, must be
	 * done AFTER configuration is complete.
	 */
	err = baresip_init(conf_config());
	if (err)
	{
		warning("main: baresip init failed (%m)\n", err);
		term();
	}

	/* Set audio path preferring the one given in -p argument (if any) */
	if (audio_path)
		play_set_path(baresip_player(), audio_path);
	else if (str_isset(conf_config()->audio.audio_path))
	{
		play_set_path(baresip_player(),
					  conf_config()->audio.audio_path);
	}

	/* NOTE: must be done after all arguments are processed */
	if (modc)
	{

		info("pre-loading modules: %zu\n", modc);

		for (i = 0; i < modc; i++)
		{

			err = module_preload(modv[i]);
			if (err)
			{
				re_fprintf(stderr,
						   "could not pre-load module"
						   " '%s' (%m)\n",
						   modv[i], err);
			}
		}
	}

	/* Initialise User Agents */
	err = ua_init(software, true, true, true);
	if (err)
		term();

	uag_set_exit_handler(ua_exit_handler, NULL);

	if (ua_eprm)
	{
		err = uag_set_extra_params(ua_eprm);
		if (err)
			term();
	}

	if (sip_trace)
		uag_enable_sip_trace(true);

	err = vidsrc_register(&myVidsrc, baresip_vidsrcl(),
						  "vidpipe", pipe_alloc, NULL);

	/* Load modules */
	err = conf_modules();
	if (err)
		term();

	if (run_daemon)
	{
		err = sys_daemon();
		if (err)
			term();

		log_enable_stdout(false);
	}

	info("baresip is ready.\n");

	/* Execute any commands from input arguments */
	for (i = 0; i < execmdc; i++)
	{
		ui_input_str(execmdv[i]);
	}

	if (tmo)
	{
		tmr_start(&tmr_quit, tmo * 1000, tmr_quit_handler, NULL);
	}

	/* Main loop */

	myThread = boost::thread(std::ref(*this));

	return true;
}

void BaresipVideoAdapter::operator()()
{
	info("Running re_main now");
	err = re_main(signal_handler);
}

bool BaresipVideoAdapter::term()
{
	tmr_cancel(&tmr_quit);

	if (err)
		ua_stop_all(true);

	ua_close();

	/* note: must be done before mod_close() */
	module_app_unload();

	conf_close();

	baresip_close();

	/* NOTE: modules must be unloaded after all application
	 * activity has stopped.
	 */
	debug("main: unloading modules..\n");
	mod_close();

	re_thread_async_close();

	/* Check for open timers */
	tmr_debug();

	libre_close();

	/* Check for memory leaks */
	free(st_ptr->buffer);
	mem_debug();

	return true;
}

bool BaresipVideoAdapter::process(void *frame_data)
{
	if (startWriting)
	{
		size_t read_size = st_ptr->vidsize.w * st_ptr->vidsize.h * 1.5;
		unsigned char *Y = st_ptr->buffer;

		st_ptr->buffer = static_cast<unsigned char *>(frame_data);

		struct timeval ts;
		uint64_t timestamp;
		struct vidframe frame;
		st_ptr->pixfmt = 0;
		vidframe_init_buf(&frame, VID_FMT_YUV420P, &st_ptr->vidsize, st_ptr->buffer);

		gettimeofday(&ts, NULL);
		timestamp = 1000000U * ts.tv_sec + ts.tv_usec;
		timestamp = timestamp * VIDEO_TIMEBASE / 1000000U;

		st_ptr->frameh(&frame, timestamp, st_ptr->arg);
	}
	return true;
}