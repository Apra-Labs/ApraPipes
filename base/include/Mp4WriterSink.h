#pragma once

#include "Module.h"

class DetailAbs;
class DetailJpeg;
class DetailH264;

class Mp4WriterSinkProps : public ModuleProps
{
public:
	Mp4WriterSinkProps(uint32_t _chunkTime, uint32_t _syncTimeInSecs, uint16_t _fps, std::string _baseFolder, bool _recordedTSBasedDTS = true, bool _enableMetadata = true, bool _useClipExportStrategy = false, int64_t _holeGapMultiplier = 3 , int64_t _minHoleGapMs = 2000, int64_t _jitterToleranceMultiplier = 5, int64_t _gapThresholdMs = 0)  : ModuleProps()
	{
		baseFolder = _baseFolder;
		fps = _fps;
		recordedTSBasedDTS = _recordedTSBasedDTS;
		enableMetadata = _enableMetadata;
		useClipExportStrategy = _useClipExportStrategy;
		holeGapMultiplier = _holeGapMultiplier;
		minHoleGapMs = _minHoleGapMs;
    	jitterToleranceMultiplier = _jitterToleranceMultiplier;
		gapThresholdMs = _gapThresholdMs;
		if ((_chunkTime >= 1 && _chunkTime <= 60) || (_chunkTime == UINT32_MAX))
		{
			chunkTime = _chunkTime;
		}
		else
		{
			throw AIPException(AIP_FATAL, "ChunkTime should be within [1,60] minutes limit or UINT32_MAX");
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
		baseFolder = "./data/Mp4_videos/";
		chunkTime = 1;
		syncTimeInSecs = 1;
		fps = 30;
		recordedTSBasedDTS = true;
		enableMetadata = true;
		useClipExportStrategy = false;
		holeGapMultiplier = 3;
		minHoleGapMs = 2000;
		jitterToleranceMultiplier = 5;
		gapThresholdMs = 0;
	}

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() +
			sizeof(recordedTSBasedDTS) +
			sizeof(baseFolder) +
			sizeof(chunkTime) +
			sizeof(syncTimeInSecs) +
			sizeof(fps) +
			sizeof(enableMetadata) +
			sizeof(useClipExportStrategy) +
			sizeof(holeGapMultiplier) +
			sizeof(minHoleGapMs) +
			sizeof(jitterToleranceMultiplier) +
			sizeof(gapThresholdMs);
	}

	std::string baseFolder;
	uint32_t chunkTime = 1;
	uint32_t syncTimeInSecs = 1;
	uint16_t fps = 30;
	bool recordedTSBasedDTS = true;
	bool enableMetadata = true;
	bool useClipExportStrategy = false;
	int64_t holeGapMultiplier = 3;
	int64_t minHoleGapMs = 2000;
	int64_t jitterToleranceMultiplier = 5;
	int64_t gapThresholdMs = 0;

private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar &boost::serialization::base_object<ModuleProps>(*this);
		ar &recordedTSBasedDTS;
		ar &baseFolder;
		ar &chunkTime;
		ar &syncTimeInSecs;
		ar &fps;
		ar &enableMetadata;
		ar &useClipExportStrategy;
		ar &holeGapMultiplier;
		ar &minHoleGapMs;
		ar &jitterToleranceMultiplier;
		ar &gapThresholdMs;
	}
};

// Error codes raised through the error callback (APErrorObject::getErrorCode())
// when writing to storage fails or recovers. Raised once per state change, not
// once per failed frame.
namespace Mp4WriterSinkErrors
{
	enum Code : int
	{
		DISK_FULL = 7001,  // no space left on the recording filesystem (ENOSPC / EDQUOT)
		DISK_ERROR = 7002, // any other storage failure (EIO, EROFS, ENOENT, EACCES, ...)
		RECOVERED = 7003   // a sync to disk succeeded after DISK_FULL / DISK_ERROR
	};
}

enum class Mp4WriteFailure
{
	NONE,      // success, or a failure that is not about storage (e.g. ENOMEM, EINVAL)
	DISK_FULL,
	DISK_ERROR
};

// Tracks the storage state of an Mp4WriterSink and decides which error code, if
// any, to raise for each write / sync result.
class Mp4WriteHealth
{
public:
	// err:        positive errno-style code of the failed call.
	// savedErrno: errno captured right after the call. libmp4 reports a failed
	//             sample write as EIO; the real cause (e.g. ENOSPC) is still in errno.
	static Mp4WriteFailure classify(int err, int savedErrno);

	// Returns the error code to raise for this failure, or 0 when the state is unchanged.
	int onFailure(Mp4WriteFailure failure);
	// Returns RECOVERED when the writer was in a failed state, otherwise 0.
	int onSuccessfulSync();
	Mp4WriteFailure state() const { return mState; }

private:
	Mp4WriteFailure mState = Mp4WriteFailure::NONE;
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
	bool doMp4MuxSync();
	bool closeCurrentOpenFile();
	uint64_t getLastWrittenFrameTS();
protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool processEOS(string &pinId);
	bool validateInputPins();
	bool validateInputOutputPins();
	bool setMetadata(framemetadata_sp &inputMetadata);
	bool handlePropsChange(frame_sp &frame);
	bool shouldTriggerSOS();
	void addInputPin(framemetadata_sp& metadata, string& pinId);
	bool enableMp4Metadata(framemetadata_sp &inputMetadata);
	bool handleCommand(Command::CommandType type, frame_sp& frame);
	void raisePendingStorageErrors();

	boost::shared_ptr<DetailAbs> mDetail;
	Mp4WriterSinkProps mProp;

};