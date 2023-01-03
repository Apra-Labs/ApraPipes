#pragma once

#include "Module.h"

class DetailAbs;
class DetailJpeg;
class DetailH264;

class Mp4WriterSinkProps : public ModuleProps
{
public:
	Mp4WriterSinkProps(uint32_t _chunkTime, uint32_t _syncTime, uint16_t _fps, std::string _baseFolder, bool _recordedTSBasedDTS = true) : ModuleProps()
	{
		baseFolder = _baseFolder;
		fps = _fps;
		recordedTSBasedDTS = _recordedTSBasedDTS;
		if ((_chunkTime >= 1 && _chunkTime <= 60) || (_chunkTime == UINT32_MAX))
		{
			chunkTime = _chunkTime;
		}
		else
		{
			throw AIPException(AIP_FATAL, "ChuntTime should be within [1,60] minutes limit");
		}
		if (_syncTime >= 1 && _syncTime <= 60)
		{
			syncTime = _syncTime;
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
		syncTime = 1;
		fps = 30;
		recordedTSBasedDTS = true;
	}

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() +
			sizeof(recordedTSBasedDTS) +
			sizeof(baseFolder) +
			sizeof(chunkTime) +
			sizeof(syncTime) +
			sizeof(fps);
	}

	std::string baseFolder;
	uint32_t chunkTime = 1;
	uint32_t syncTime = 1;
	uint16_t fps = 30;
	bool recordedTSBasedDTS = true;
private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar &boost::serialization::base_object<ModuleProps>(*this);
		ar &recordedTSBasedDTS;
		ar &baseFolder;
		ar &chunkTime;
		ar &syncTime;
		ar &fps;
	}
};

class Mp4WriterSink : public Module
{
public:
	Mp4WriterSink(Mp4WriterSinkProps _props);
	virtual ~Mp4WriterSink();
	bool init();
	bool term();
	void setProps(Mp4WriterSinkProps &props);
	Mp4WriterSinkProps getProps();
protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool processEOS(string &pinId);
	bool validateInputPins();
	bool validateInputOutputPins();
	bool setMetadata(framemetadata_sp &inputMetadata);
	bool handlePropsChange(frame_sp &frame);
	bool shouldTriggerSOS();
	boost::shared_ptr<DetailAbs> mDetail;
	Mp4WriterSinkProps mProp;

};