/*
 * V4LSource.cpp
 *
 *  Created on: 30-Mar-2022
 *      Author: developer
 */
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <stdio.h>
#include <getopt.h>             /* getopt_long() */
#include <fcntl.h>              /* low-level i/o */
#include <string.h>
#include <unistd.h>
#include <linux/videodev2.h>
#include "V4L2CameraSourceHelper.h"


#define CLEAR(x) memset(&(x), 0, sizeof(x))

V4L2CameraSourceHelper::V4L2CameraSourceHelper(string devPath) :
		m_debug(false), m_fd(-1), m_buffers(NULL), m_nBuffers(0) {
	m_devPath = devPath;
	// TODO Auto-generated constructor stub

}

V4L2CameraSourceHelper::~V4L2CameraSourceHelper() {
	// TODO Auto-generated destructor stub
	if (m_buffers) {
		deinitCamera();
	}
}

bool V4L2CameraSourceHelper::openCamera() {
	struct stat st;

	if (-1 == stat(m_devPath.c_str(), &st)) {
		printf("Cannot identify '%s': %d, %s\n", m_devPath.c_str(),
		errno, strerror(errno));
		return false;
	}

	if (!S_ISCHR(st.st_mode)) {
		printf("%s is no device\n", m_devPath.c_str());
		return false;
	}

	m_fd = open(m_devPath.c_str(), O_RDWR /* required */| O_NONBLOCK, 0);

	if (-1 == m_fd) {
		printf("Cannot open '%s': %d, %s\n", m_devPath.c_str(), errno,
				strerror(errno));
		return false;
	}
	return true;
}

bool V4L2CameraSourceHelper::initCamera() {

	struct v4l2_capability cap;
	struct v4l2_cropcap cropcap;
	struct v4l2_crop crop;
	struct v4l2_format fmt;

	if (-1 == xioctl(m_fd, VIDIOC_QUERYCAP, &cap)) {
		if (m_debug) {
			if (EINVAL == errno) {
				printf("%s is no V4L2 device\n", m_devPath.c_str());
			} else {
				printf("error in VIDIOC_QUERYCAP\n");
			}
		}
		return false;
	}

	if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
		printf("%s is no video capture device\n", m_devPath.c_str());
		return false;
	}

	/*
	 switch (io) {
	 case IO_METHOD_READ:
	 if (!(cap.capabilities & V4L2_CAP_READWRITE)) {
	 fprintf(stderr, "%s does not support read i/o\n", m_devPath.c_str());
	 exit(EXIT_FAILURE);
	 }
	 break;

	 case IO_METHOD_MMAP:
	 case IO_METHOD_USERPTR:
	 if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
	 fprintf(stderr, "%s does not support streaming i/o\n", m_devPath.c_str());
	 exit(EXIT_FAILURE);
	 }
	 break;
	 }
	 */
	if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
		printf("%s does not support streaming i/o\n", m_devPath.c_str());
		return false;
	}

	/* Select video input, video standard and tune here. */

	CLEAR(cropcap);

	cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

	if (0 == xioctl(m_fd, VIDIOC_CROPCAP, &cropcap)) {
		crop.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		crop.c = cropcap.defrect; /* reset to default */

		if (-1 == xioctl(m_fd, VIDIOC_S_CROP, &crop)) {
			switch (errno) {
			case EINVAL:
				/* Cropping not supported. */
				break;
			default:
				/* Errors ignored. */
				break;
			}
		}
	} else {
		/* Errors ignored. */
	}

	CLEAR(fmt);
//	fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
//	if (-1 == xioctl(m_fd, VIDIOC_G_FMT, &fmt)){
//		printf("error:: VIDIOC_G_FMT\n");
//		return false;
//	}

	return true;
}

bool V4L2CameraSourceHelper::requestBuffers(uint16_t bufferCount) {
	struct v4l2_requestbuffers req;

	CLEAR(req);

	req.count = bufferCount;
	req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	req.memory = V4L2_MEMORY_MMAP;

	if (-1 == xioctl(m_fd, VIDIOC_REQBUFS, &req)) {
		if (EINVAL == errno) {
			printf("%s does not support "
					"memory mapping\n", m_devPath.c_str());
		} else {
			printf("error:: VIDIOC_REQBUFS\n");
		}
		return false;
	}

	if (req.count < 2) {
		printf("Insufficient buffer memory on %s\n", m_devPath.c_str());
		return false;
	}

	m_buffers = (buffer*) calloc(req.count, sizeof(*m_buffers));

	if (!m_buffers) {
		printf("Out of memory\n");
		return false;
	}

	for (m_nBuffers = 0; m_nBuffers < req.count; ++m_nBuffers) {
		struct v4l2_buffer buf;

		CLEAR(buf);

		buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf.memory = V4L2_MEMORY_MMAP;
		buf.index = m_nBuffers;

		if (-1 == xioctl(m_fd, VIDIOC_QUERYBUF, &buf)) {
			printf("error:: VIDIOC_QUERYBUF\n");
			return false;
		}

		m_buffers[m_nBuffers].length = buf.length;
		m_buffers[m_nBuffers].start = mmap(NULL /* start anywhere */,
				buf.length,
				PROT_READ | PROT_WRITE /* required */,
				MAP_SHARED /* recommended */, m_fd, buf.m.offset);

		if (MAP_FAILED == m_buffers[m_nBuffers].start) {
			printf("error:: mmap\n");
			return false;
		}
	}
	return true;
}

bool V4L2CameraSourceHelper::startStreaming() {
	for (int i = 0; i < m_nBuffers; ++i) {
		struct v4l2_buffer buf;

		CLEAR(buf);
		buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf.memory = V4L2_MEMORY_MMAP;
		buf.index = i;

		if (-1 == xioctl(m_fd, VIDIOC_QBUF, &buf)) {
			printf("error:: VIDIOC_QBUF\n");
			return false;
		}
	}
	v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	if (-1 == xioctl(m_fd, VIDIOC_STREAMON, &type)) {
		printf("error:: VIDIOC_STREAMON\n");
		return false;
	}
	return true;
}

bool V4L2CameraSourceHelper::readFrame(uint8_t *frameBuffer, uint64_t &frameBufferSize) {
	struct v4l2_buffer buf;
	fd_set fds;
	struct timeval tv;
	int r;

	FD_ZERO(&fds);
	FD_SET(m_fd, &fds);

	/* Timeout. */
	tv.tv_sec = 20;
	tv.tv_usec = 0;
	bool isError = true;

	for (int i = 0; i < 3; i++) {
		r = select(m_fd + 1, &fds, NULL, NULL, &tv);

		if (-1 == r) {
			if (EINTR == errno)
				continue;
			printf("error:: select\n");
			return false;
		}

		if (0 == r) {
			printf("error:: select timeout\n");
			return false;
		}
		isError = false;
		break;
	}
	if (isError) {
		printf("error:: EINTR\n");
		return false;
	}
	if (!dequeBuffer(buf)) {
		return false;
	}
	frameBufferSize = buf.bytesused;
	memcpy(frameBuffer, m_buffers[buf.index].start, buf.bytesused);
	return enqueBuffer(buf);
}

bool V4L2CameraSourceHelper::enqueBuffer(v4l2_buffer &buf) {
	if (-1 == xioctl(m_fd, VIDIOC_QBUF, &buf)) {
		printf("error:: VIDIOC_QBUF\n");
		return false;
	}
	return true;
}
bool V4L2CameraSourceHelper::dequeBuffer(v4l2_buffer &buf) {
	CLEAR(buf);

	buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	buf.memory = V4L2_MEMORY_MMAP;

	if (-1 == xioctl(m_fd, VIDIOC_DQBUF, &buf)) {
		switch (errno) {
		case EAGAIN:
			printf("error:: VIDIOC_DQBUF: EAGAIN\n");
			return 0;
		case EIO:
			/* Could ignore EIO, see spec. */

			/* fall through */

		default:
			printf("error:: VIDIOC_DQBUF\n");
			return false;
		}
	}
	return true;
}

bool V4L2CameraSourceHelper::stopStreaming() {
	v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	if (-1 == xioctl(m_fd, VIDIOC_STREAMOFF, &type)) {
		printf("error:: VIDIOC_STREAMOFF\n");
		return false;
	}
	return true;
}

bool V4L2CameraSourceHelper::deinitCamera() {
	for (uint16_t i = 0; i < m_nBuffers; ++i) {
		if (-1 == munmap(m_buffers[i].start, m_buffers[i].length)) {
			printf("error:: munmap\n");
			return false;
		}
	}
	m_buffers = NULL;
	m_nBuffers = 0;
	return true;
}

bool V4L2CameraSourceHelper::closeCamera() {
	if (-1 == close(m_fd)) {
		printf("error:: close");
		return false;
	}
	m_fd = -1;
	return true;
}

int V4L2CameraSourceHelper::xioctl(int fh, int request, void *arg) {
	int r;
	do {
		r = ioctl(fh, request, arg);
	} while (-1 == r && EINTR == errno);

	return r;
}

bool V4L2CameraSourceHelper::setFormatBGGR10(uint32_t width, uint32_t height,
		v4l2_format &fmt) {
//	unsigned int min;
	fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	fmt.fmt.pix.width = width; //replace
	fmt.fmt.pix.height = height; //replace
	fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_SBGGR10; //replace
	fmt.fmt.pix.field = V4L2_FIELD_ANY;
	fmt.fmt.pix.bytesperline = fmt.fmt.pix.width * 2;
	fmt.fmt.pix.sizeimage = fmt.fmt.pix.bytesperline * fmt.fmt.pix.height;
//	fmt.fmt.pix.field = V4L2_FIELD_NONE;

	if (-1 == xioctl(m_fd, VIDIOC_S_FMT, &fmt)) {
		printf("Could not set format error:: VIDIOC_S_FMT\n");
		return false;
	}

	if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_SBGGR10) {
		printf("Libv4l didn't accept RGB24 format. Can't proceed.\n");
		return false;;
	}
	if ((fmt.fmt.pix.width != width) || (fmt.fmt.pix.height != height))
		printf("Warning: driver is sending image at %dx%d\n", fmt.fmt.pix.width,
				fmt.fmt.pix.height);

	return true;
}

bool V4L2CameraSourceHelper::setFormatRGGB10(uint32_t width, uint32_t height,
		v4l2_format &fmt) {
	unsigned int min;
	fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	fmt.fmt.pix.width = width; //replace
	fmt.fmt.pix.height = height; //replace
	fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_SRGGB10; //replace
	fmt.fmt.pix.field = V4L2_FIELD_NONE;
	fmt.fmt.pix.bytesperline = fmt.fmt.pix.width * 2;
	fmt.fmt.pix.sizeimage = fmt.fmt.pix.bytesperline * fmt.fmt.pix.height;

	if (-1 == xioctl(m_fd, VIDIOC_S_FMT, &fmt)) {
		printf("Could not set format error:: VIDIOC_S_FMT\n");
		return false;
	}

	if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_SRGGB10) {
		printf("Libv4l didn't accept RGB24 format. Can't proceed.\n");
		return false;;
	}
	if ((fmt.fmt.pix.width != width) || (fmt.fmt.pix.height != height))
		printf("Warning: driver is sending image at %dx%d\n", fmt.fmt.pix.width,
				fmt.fmt.pix.height);
	return true;
}

