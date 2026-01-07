#pragma once

#include "Module.h"
#include <map>
#include <variant>

class MultimediaQueueXform;
class MultimediaQueueXformProps : public ModuleProps
{
public:
	// Note: Removed ambiguous default constructor - use constructor with defaults below
	MultimediaQueueXformProps(uint32_t queueLength = 10000, uint16_t tolerance = 5000, int _mmqFps = 24, bool _isDelayTime = true)
	{
		lowerWaterMark = queueLength;
		upperWaterMark = queueLength + tolerance;
		isMapDelayInTime = _isDelayTime;
		mmqFps = _mmqFps;
	}

	uint32_t lowerWaterMark; // Length of multimedia queue in terms of time or number of frames
	uint32_t upperWaterMark; //Length of the multimedia queue when the next module queue is full
	bool isMapDelayInTime;
	int mmqFps;

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(lowerWaterMark) + sizeof(upperWaterMark) + sizeof(isMapDelayInTime) + sizeof(mmqFps);
	}

	int startIndex;
	int maxIndex;
	string strFullFileNameWithPattern;
	bool readLoop;

private:
	friend class boost::serialization::access;

	template<class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar & boost::serialization::base_object<ModuleProps>(*this);
		ar & lowerWaterMark;
		ar & upperWaterMark;
		ar & isMapDelayInTime;
		ar & mmqFps;
	}

public:
	// Declarative pipeline property binding
	void applyProperties(const std::map<std::string, std::variant<int64_t, double, bool, std::string>>& props)
	{
		for (const auto& [key, value] : props) {
			if (key == "queueLength" || key == "lowerWaterMark") {
				if (std::holds_alternative<int64_t>(value)) {
					lowerWaterMark = static_cast<uint32_t>(std::get<int64_t>(value));
				}
			} else if (key == "tolerance") {
				if (std::holds_alternative<int64_t>(value)) {
					upperWaterMark = lowerWaterMark + static_cast<uint32_t>(std::get<int64_t>(value));
				}
			} else if (key == "upperWaterMark") {
				if (std::holds_alternative<int64_t>(value)) {
					upperWaterMark = static_cast<uint32_t>(std::get<int64_t>(value));
				}
			} else if (key == "mmqFps") {
				if (std::holds_alternative<int64_t>(value)) {
					mmqFps = static_cast<int>(std::get<int64_t>(value));
				}
			} else if (key == "isMapDelayInTime") {
				if (std::holds_alternative<bool>(value)) {
					isMapDelayInTime = std::get<bool>(value);
				}
			}
		}
	}

	std::variant<int64_t, double, bool, std::string> getProperty(const std::string& name) const
	{
		if (name == "queueLength" || name == "lowerWaterMark") return static_cast<int64_t>(lowerWaterMark);
		if (name == "upperWaterMark") return static_cast<int64_t>(upperWaterMark);
		if (name == "mmqFps") return static_cast<int64_t>(mmqFps);
		if (name == "isMapDelayInTime") return isMapDelayInTime;
		return int64_t(0);
	}
};

class State;

class MultimediaQueueXform : public Module {
public:
	MultimediaQueueXform(MultimediaQueueXformProps _props);

	virtual ~MultimediaQueueXform() {
	}
	bool init();
	bool term();
	void setState(uint64_t ts, uint64_t te);
	bool handleCommand(Command::CommandType type, frame_sp& frame);
	bool allowFrames(uint64_t& ts, uint64_t& te);
	void addInputPin(framemetadata_sp& metadata, string& pinId);
	void setProps(MultimediaQueueXformProps _props);
	MultimediaQueueXformProps getProps();
	bool handlePropsChange(frame_sp& frame);
	boost::shared_ptr<State> mState;
	MultimediaQueueXformProps mProps;
	boost::shared_ptr<FrameContainerQueue> getQue();
	void extractFramesAndEnqueue(boost::shared_ptr<FrameContainerQueue>& FrameQueue);
	void setMmqFps(int fps);
	void setPlaybackSpeed(float playbackSpeed);
	void stopExportFrames();
protected:
	bool process(frame_container& frames);
	bool validateInputPins();
	bool validateOutputPins();
	bool validateInputOutputPins();

private:
	void getQueueBoundaryTS(uint64_t& tOld, uint64_t& tNew);
	std::string mOutputPinId;
	bool pushToNextModule = true;
	bool reset = false;
	uint64_t startTimeSaved = 0;
	uint64_t endTimeSaved = 0;
	uint64_t queryStartTime = 0;
	uint64_t queryEndTime = 0;
	bool direction = true;
	FrameMetadata::FrameType mFrameType;
	using sys_clock = std::chrono::system_clock;
	sys_clock::time_point frame_begin;
	std::chrono::nanoseconds myTargetFrameLen;
	std::chrono::nanoseconds myNextWait;
	uint64_t latestFrameExportedFromHandleCmd = 0;
	uint64_t latestFrameExportedFromProcess = 0;
	bool initDone = false;
	int framesToSkip = 0;
	int initialFps = 0;
	float speed = 1;
	bool exportFrames;
};
