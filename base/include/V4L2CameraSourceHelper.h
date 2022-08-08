/*
 * V4LSource.h
 *
 *  Created on: 30-Mar-2022
 *      Author: developer
 */

#ifndef V4L2SOURCE_H_
#define V4L2SOURCE_H_
#include <string>
#include <linux/videodev2.h>

using namespace std;

struct buffer {
	void *start;
	size_t length;
};

class V4L2CameraSourceHelper {
public:
	V4L2CameraSourceHelper(string devPath);
	virtual ~V4L2CameraSourceHelper();
	bool openCamera();
	bool initCamera();
	bool setFormatBGGR10(uint32_t width, uint32_t height, v4l2_format &fmt);
	bool setFormatRGGB10(uint32_t width, uint32_t height, v4l2_format &fmt);
	bool requestBuffers(uint16_t bufferCount);
	bool startStreaming();
	bool dequeBuffer(v4l2_buffer &buf); //	if readFrame is used, do not use this func
	bool readFrame(uint8_t *frameBuffer, uint64_t &size);
	bool enqueBuffer(v4l2_buffer &buf); //	if readFrame is used, do not use this func
	bool stopStreaming();
	bool deinitCamera();
	bool closeCamera();
private:
	int xioctl(int fh, int request, void *arg);
	bool m_debug;
	int m_fd;
	string m_devPath;
	buffer *m_buffers;
	uint16_t m_nBuffers;

};

#endif /* V4L2SOURCE_H_ */
