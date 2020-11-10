#include "GPIODriver.h"
#include <fcntl.h>

#include "Logger.h"

#define SYS_GPIO_PATH "/sys/class/gpio"
#define GPIO_TOUT_USEC 100000

GPIODriver::GPIODriver(uint32_t gpioNo) : m_fd(-1), m_gpio(gpioNo), m_mSecTout(0)
{
}

GPIODriver::~GPIODriver()
{
	Close();
}

bool GPIODriver::Init(bool isRead)
{
	m_isRead = isRead;
	UnExport();
	Export();

	if (!SetDirection(isRead))
	{
		LOG_ERROR << "SET DIRECTION FAILED READ<>" << isRead << " " << m_gpio;
		return false;
	}

	return Open();
}

bool GPIODriver::Init4EdgeInterrupt(unsigned long mSecTout)
{
	if (!Init(true))
	{
		LOG_ERROR << "Init4EdgeInterrupt Init FAILED " << m_gpio;
		return false;
	}

	m_fdset.fd = m_fd;
	m_fdset.events = POLLPRI;
	m_mSecTout = mSecTout;

	return SetGPIOEdgeEvent();
}

bool GPIODriver::SetGPIOEdgeEvent()
{
	int fd;
	char buff[64] = {0};

	snprintf(buff, sizeof(buff), SYS_GPIO_PATH "/gpio%d/edge", m_gpio);

	fd = open(buff, O_WRONLY);
	if (fd < 0)
	{
		LOG_ERROR << "Unable to open gpio%d/edge " << m_gpio;
		return false;
	}

	std::string edgeStr = "both";
	if (write(fd, edgeStr.c_str(), edgeStr.length() + 1) != ((int)(edgeStr.length() + 1)))
	{
		LOG_ERROR << "Error setting edge to " << edgeStr.c_str() <<  " " << m_gpio;
		return false;
	}
	close(fd);
	return true;
}

int GPIODriver::Open()
{
	char buff[64] = {0};
	snprintf(buff, sizeof(buff), SYS_GPIO_PATH "/gpio%d/value", m_gpio);

	m_fd = open(buff, (m_isRead ? (O_RDONLY | O_NONBLOCK) : O_WRONLY));
	if (m_fd < 0)
	{
		LOG_ERROR << "Error opening gpio " << m_gpio;
		return -1;
	}
	return m_fd;
}

bool GPIODriver::Close()
{
	if (m_fd >= 0)
	{
		close(m_fd);
		m_fd = -1;
	}

	return true;
}

bool GPIODriver::Export()
{
	int fd, length;
	char buff[64] = {0};

	fd = open(SYS_GPIO_PATH "/export", O_WRONLY);
	if (fd < 0)
	{
		LOG_ERROR << "unable to open export for gpio " << m_gpio;
		return false;
	}

	length = snprintf(buff, sizeof(buff), "%d", m_gpio);
	if (write(fd, buff, length) != length)
	{
		LOG_ERROR << "unable to export gpio " << m_gpio;
		return false;
	}
	close(fd);
	usleep(GPIO_TOUT_USEC);
	return true;
}

bool GPIODriver::UnExport()
{
	int fd, length;
	char buff[64] = {0};

	fd = open(SYS_GPIO_PATH "/unexport", O_WRONLY);
	if (fd < 0)
	{
		LOG_ERROR << "unable to open gpio for unexport " << m_gpio;
		return false;
	}

	length = snprintf(buff, sizeof(buff), "%d", m_gpio);
	if (write(fd, buff, length) != length)
	{
		LOG_ERROR << "unable to unexport gpio " << m_gpio;
		return false;
	}
	close(fd);
	return true;
}

bool GPIODriver::SetDirection(bool isRead)
{
	int fd;
	char buff[64] = {0};

	snprintf(buff, sizeof(buff),
			 SYS_GPIO_PATH "/gpio%d/direction", m_gpio);

	fd = open(buff, O_WRONLY);
	if (fd < 0)
	{
		LOG_ERROR << "SetDirection unable to open gpio " << m_gpio;
		return false;
	}

	if (!isRead)
	{
		if (write(fd, "out", 4) != 4)
		{
			LOG_ERROR << "unable to make gpio as output " << m_gpio;
			return false;
		}
	}
	else
	{
		if (write(fd, "in", 3) != 3)
		{
			LOG_ERROR << "unable to make gpio as input " << m_gpio;
			return false;
		}
	}
	close(fd);
	return true;
}

int GPIODriver::Read()
{
	if (m_fd < 0)
	{
		return -1;
	}
	lseek(m_fd, 0, SEEK_SET);

	char ch;
	if (read(m_fd, &ch, 1) != 1)
	{
		LOG_ERROR << "Error fetching GPIODriver value " << m_gpio;
		return -1;
	}

	if (ch != '0')
	{
		return 1;
	}
	return 0;
}

int GPIODriver::ReadWithInterrupt()
{
	auto rc = poll(&m_fdset, 1, m_mSecTout);
	if (rc < 0)
	{
		LOG_ERROR << "poll() failed " << m_gpio;
		return -1;
	}
	else if (rc == 0)
	{
		return -1;
	}

	if (m_fdset.revents & POLLPRI)
	{
		return Read();
	}

	return -1;
}

bool GPIODriver::Write(bool makeHigh)
{
	if (m_fd < 0)
	{
		return false;
	}
	std::string state = makeHigh ? "1" : "0";
	if (write(m_fd, state.c_str(), 2) != 2)
	{
		LOG_ERROR << "Error setting GPIODriver state " << state.c_str() << " " << m_gpio;
		return false;
	}
	return true;
}
