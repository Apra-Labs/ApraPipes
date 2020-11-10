#include "GPIOSource.h"
#include "GPIOMetadata.h"
#include "GPIODriver.h"
#include "Frame.h"
#include <chrono>

using sys_clock = std::chrono::system_clock;

class GPIOSource::Detail
{
public:
	Detail(uint32_t highTime) : prevValue(-1), sendAlways(false)
	{
		if (highTime == 0)
		{
			sendAlways = true;
		}
		else
		{
			mHighTime = std::chrono::nanoseconds(highTime * 1000 * 1000); // ms -> ns
			mMaxHighTime = std::chrono::nanoseconds(2 * highTime * 1000 * 1000);
			prevValueTS = sys_clock::now();
		}
	}

	void setInitialValue(int value)
	{
		prevValue = value;
	}

	bool shouldSend(int& value)
	{
		if (sendAlways)
		{
			return true;
		}

		bool res = false;

		if (prevValue == 0 && value == 1)
		{
			res = true;
		}

		prevValue = value;
		prevValueTS = sys_clock::now();

		return res;
	}

	~Detail() {}

private:
	bool sendAlways;
	std::chrono::nanoseconds mHighTime;
	std::chrono::nanoseconds mMaxHighTime;
	int prevValue;

	sys_clock::time_point prevValueTS;
};

GPIOSource::GPIOSource(GPIOSourceProps _props)
	: Module(SOURCE, "GPIOSource", _props), mProps(_props)
{
	mOutputMetadata = framemetadata_sp(new GPIOMetadata());
	mOutputPinId = addOutputPin(mOutputMetadata);
	mOutputSize = mOutputMetadata->getDataSize();

	mDetail.reset(new Detail(_props.highTime));
}

GPIOSource::~GPIOSource() {}

bool GPIOSource::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{
		return false;
	}

	return true;
}

bool GPIOSource::init()
{
	if (!Module::init())
	{
		return false;
	}

	mDriver.reset(new GPIODriver(mProps.gpioNo));

	return mDriver->Init4EdgeInterrupt(100);
}

bool GPIOSource::term()
{
	mDriver.reset();

	return Module::term();
}

void GPIOSource::notifyPlay(bool play)
{
	if (!play)
	{
		return;
	}

	// on every play - reading the initial value
	auto value = mDriver->Read();
	mDetail->setInitialValue(value);

	sendValue(value);
}

bool GPIOSource::produce()
{
	auto value = mDriver->ReadWithInterrupt();
	if (value == -1)
	{
		return true;
	}

	return sendValue(value);
}

bool GPIOSource::sendValue(int value)
{
	if (!mDetail->shouldSend(value))
	{
		return true;
	}
	// send frame
	auto frame = makeFrame(mOutputSize, mOutputMetadata);
	auto buffer = static_cast<unsigned char *>(frame->data());
	buffer[0] = static_cast<unsigned char>(value);

	frame_container frames;
	frames.insert(make_pair(mOutputPinId, frame));
	send(frames);

	return true;
}