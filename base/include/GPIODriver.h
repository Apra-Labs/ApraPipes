#pragma once

#include <stdint.h>
#include <linux/input.h>
#include <poll.h>

class GPIODriver
{
public:
	GPIODriver(uint32_t gpioNo);
	virtual ~GPIODriver();
	bool Init(bool isRead);
	bool Init4EdgeInterrupt(unsigned long mSecTout);
	int Read();
	int ReadWithInterrupt();
	bool Write(bool makeHigh);

private:
	int Open();
	bool Close();
	bool Export();
	bool SetDirection(bool isRead);
	bool UnExport();
	bool SetGPIOEdgeEvent();

	int m_fd;
	uint32_t m_gpio;
	bool m_isRead;

	struct pollfd m_fdset;
	unsigned long m_mSecTout;
};
