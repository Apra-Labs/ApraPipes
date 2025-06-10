#pragma once

#include "Module.h"

class DetailAbs;
class DetailJpeg;
class DetailH264;

using CallbackFunction = std::function<void()>;

class Mp4WriterSinkProps : public ModuleProps
{
public:
	Mp4WriterSinkProps(uint32_t _chunkTime, uint32_t _syncTimeInSecs, uint16_t _fps, std::string _baseFolder) : ModuleProps()
	{
		baseFolder = _baseFolder;
		fps = _fps;
		if ((_chunkTime >= 1 && _chunkTime <= 60) || (_chunkTime == UINT32_MAX))
		{
			chunkTime = _chunkTime;
		}
		else
		{
			throw AIPException(AIP_FATAL, "ChuntTime should be within [1,60] minutes limit or UINT32_MAX");
		}
		if (_syncTimeInSecs >= 1 && _syncTimeInSecs <= 60)
		{
			syncTimeInSecs = _syncTimeInSecs;
		}
		else
		{
			throw AIPException(AIP_FATAL, "SyncTime should be within [1,60] minutes limit");
		}
	}

	Mp4WriterSinkProps() : ModuleProps()
	{
		baseFolder = "./data/mp4_videos/";
		chunkTime = 1; //minutes
		syncTimeInSecs = 1;
		fps = 30;
	}

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() +
			sizeof(baseFolder) +
			sizeof(chunkTime) +
			sizeof(syncTimeInSecs) +
			sizeof(fps);
	}

	std::string baseFolder;
	uint32_t chunkTime = 1;
	uint32_t syncTimeInSecs = 1;
	uint16_t fps = 30;
private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar &boost::serialization::base_object<ModuleProps>(*this);
		ar &baseFolder;
		ar &chunkTime;
		ar &syncTimeInSecs;
		ar &fps;
	}
};

class Mp4WriterSink : public Module
{
public:
	Mp4WriterSink(Mp4WriterSinkProps _props);
	virtual ~Mp4WriterSink();
	void registerCallback(const CallbackFunction &_callback)
	{
		m_callbackFunction = _callback;
	}
	bool init();
	bool term();
	void setProps(Mp4WriterSinkProps &props);
	Mp4WriterSinkProps getProps();
	bool closeFile();
	void setCustomMetadata(std::string data);
	bool retortCallback();
	std::vector<std::vector<uint8_t>> getQueuedFrames();
	void hashing(uint8_t* frame, size_t frameSize);
	void hashing();
	bool isFileWriteComplete();
protected:
    bool process(frame_container &frames);
    void saveInCache(frame_container &frames);
    bool processSOS(frame_sp& frame);
	bool processEOS(string &pinId);
	bool validateInputPins();
	bool validateInputOutputPins();
	bool setMetadata(framemetadata_sp &inputMetadata);
	bool handlePropsChange(frame_sp &frame);
	bool handleCommand(Command::CommandType type, frame_sp& frame);
	bool shouldTriggerSOS();
	void cacheFrames(uint32_t firstLimit, uint32_t endLimit, frame_sp frame);
	vector<uint8_t> getFrameBytes(frame_sp frame);

	boost::shared_ptr<DetailAbs> mDetail;
	Mp4WriterSinkProps mProp;

	std::vector<std::vector<uint8_t>> m_hashFrameStartQueue;
	std::vector<std::vector<uint8_t>> m_hashFrameEndQueue;
	std::vector<std::vector<uint8_t>> m_hashFrameQueue;
	int64_t m_lastFrameStored = -1;
	int64_t m_currFrame = -1;

private:
	CallbackFunction m_callbackFunction = NULL;
	std::string m_customMetadata;
	bool m_shouldStopFileWrite;
	string m_prevFile;

	std::vector<uint8_t> m_sps;
	std::vector<uint8_t> m_pps;


};