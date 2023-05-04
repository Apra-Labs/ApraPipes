/**
* @file vidpipe.c Video source from pipeline module
*
*
*/
#include <re.h>
#include <baresip.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <rem.h>




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


//destructor
static void destructor(void *arg)
{
   struct vidsrc_st *st = arg;
   debug("vidpipe: stopping video source..\n");
   if (st->isRunning) {
       st->isRunning = false;
       thrd_join(st->thread, NULL);
   }
}


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


   // Close the file and free the buffer


   //free(buffer);
}




//Read thread


static int read_thread(void *arg)
{
   struct vidsrc_st *st = arg;
   int err;


   while (st->isRunning) {
       err = read_frame(st);
       if (err) {
           info("vidpipe: reading frame: %m\n", err);
       }
   }


   return 0;
}




//allocate video source state
static int pipe_alloc(struct vidsrc_st **stp, const struct vidsrc *vs,
        struct vidsrc_prm *prm,
        const struct vidsz *size, const char *fmt,
        const char *dev, vidsrc_frame_h *frameh,
        vidsrc_packet_h  *packeth,
        vidsrc_error_h *errorh, void *arg)
{
   struct vidsrc_st *st;
   int err;


   (void)prm;
   (void)fmt;
   (void)packeth;
   (void)errorh;


   if (!stp || !size || !frameh)
       return EINVAL;


   st = mem_zalloc(sizeof(*st), destructor);
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




//init
static int vidpipe_init(void)
{
   info("vidpipe: vidpipe video source initializing \n");
   int err;




   err = vidsrc_register(&vidsrc, baresip_vidsrcl(),
                  "vidpipe", pipe_alloc, NULL);
   if (err)
       return err;
   //list_init(&vidsrc->dev_list);
  
   return 0;
}


//close
static int vidpipe_close(void)
{
   info("vidpipe: closing\n");
   vidsrc = mem_deref(vidsrc);
   return 0;
}


//mod_export
const struct mod_export DECL_EXPORTS(foo) = {
   "vidpipe",
   "vidsrc",
   vidpipe_init,
   vidpipe_close
};


