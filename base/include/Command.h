#pragma once

#include "Utils.h"

class Command
{
public:
	enum CommandType
	{
		None,
		FileReaderModule,
		Relay,
		Step,
		ValvePassThrough,
		MultimediaQueueXform,
		Seek,
		DeleteWindow,
		CreateWindow,
		PlayPause,
		NVRCommandRecord,
		NVRCommandExport,
		NVRCommandExportMMQ,
		NVRCommandView,
		NVRGoLive,
		NVRCommandExportView,
		MP4WriterLastTS,
		MMQtimestamps,
		Rendertimestamp
	};

	Command()
	{
		type = CommandType::None;
	}

	Command(CommandType _type)
	{
		type = _type;
	}

	size_t getSerializeSize()
	{
		return 1024 + sizeof(type);
	}

	CommandType getType()
	{
		return type;
	} 

private:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int /* file_version */) {
		ar & type;
	}

	CommandType type;
};

/*
NoneCommand was introduced for getCommandType to work
boost::serialization was adding some extra bytes for child class
*/
class NoneCommand : public Command
{
public:
	NoneCommand() : Command(CommandType::None)
	{

	}

	static Command::CommandType getCommandType(void* buffer, size_t size)
	{
		NoneCommand cmd;
		Utils::deSerialize(cmd, buffer, size);

		return cmd.getType();
	}

private:

	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int /* file_version */) {
		ar & boost::serialization::base_object<Command>(*this);
	}
};

class FileReaderModuleCommand : public Command
{
public:
	FileReaderModuleCommand() : Command(CommandType::FileReaderModule)
	{
		currentIndex = 0;
	}

	FileReaderModuleCommand(uint64_t index) : Command(CommandType::FileReaderModule)
	{
		currentIndex = index;
	}

	size_t getSerializeSize()
	{
		return Command::getSerializeSize() + sizeof(currentIndex);
	}

	uint64_t getCurrentIndex()
	{
		return currentIndex;
	}

private:

	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int /* file_version */) {
		ar & boost::serialization::base_object<Command>(*this);

		ar & currentIndex;
	}

	uint64_t currentIndex;
};

class RelayCommand : public Command
{
public:
	RelayCommand() : Command(CommandType::Relay)
	{
		nextModuleId = "";
		open = true;
	}

	RelayCommand(std::string& _nextModuleId, bool _open) : Command(CommandType::Relay)
	{
		nextModuleId = _nextModuleId;
		open = _open;
	}

	size_t getSerializeSize()
	{
		return Command::getSerializeSize() + sizeof(nextModuleId) + nextModuleId.length() + sizeof(open) + 1024;
	}

	std::string nextModuleId;
	bool open;

private:

	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int /* file_version */) {
		ar & boost::serialization::base_object<Command>(*this);

		ar & nextModuleId & open;
	}

 
};

class StepCommand : public Command
{
public:
	StepCommand() : Command(CommandType::Step)
	{

	}
  
	size_t getSerializeSize()
	{
		return Command::getSerializeSize();
	}

private:

	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int /* file_version */) {
		ar & boost::serialization::base_object<Command>(*this);  
	}


};

class ValvePassThroughCommand : public Command
{
public:
	ValvePassThroughCommand() : Command(Command::CommandType::ValvePassThrough)
	{
	}

	size_t getSerializeSize()
	{
		return Command::getSerializeSize() + sizeof(numOfFrames);
	}

	int numOfFrames;

private:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int /* file_version */)
	{
		ar& boost::serialization::base_object<Command>(*this);
		ar& numOfFrames;

	}
};

class EglRendererCloseWindow : public Command
{
public:
	EglRendererCloseWindow() : Command(Command::CommandType::DeleteWindow)
	{
	}

	size_t getSerializeSize()
	{
		return Command::getSerializeSize();
	}


private:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int /* file_version */)
	{
		ar& boost::serialization::base_object<Command>(*this);
	}
};

class EglRendererCreateWindow : public Command
{
public:
	EglRendererCreateWindow() : Command(Command::CommandType::CreateWindow)
	{
	}

	size_t getSerializeSize()
	{
		return Command::getSerializeSize() + sizeof(width) + sizeof(height) ;
	}
	int width;
	int height;

private:
	friend class boost::serialization::access;
	template <class Archive>
	void serialize(Archive &ar, const unsigned int /* file_version */)
	{
		ar &boost::serialization::base_object<Command>(*this);
		ar &width;
		ar &height;
	}
};

class MultimediaQueueXformCommand : public Command
{
public:
	MultimediaQueueXformCommand() : Command(Command::CommandType::MultimediaQueueXform)
	{
	}

	size_t getSerializeSize()
	{
		return Command::getSerializeSize() + sizeof(startTime) + sizeof(endTime);
	}

	int64_t startTime = 0;
	int64_t endTime = 0;

private:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int /* file_version */)
	{
		ar& boost::serialization::base_object<Command>(*this);
		ar& startTime;
		ar& endTime;
	}
};


class Mp4SeekCommand : public Command
{
public:
	Mp4SeekCommand() : Command(CommandType::Seek)
	{

	}

	Mp4SeekCommand(uint64_t _skipTS, bool _forceReopen = false) : Command(CommandType::Seek)
	{
		seekStartTS = _skipTS;
		forceReopen = _forceReopen;
	}

	size_t getSerializeSize()
	{
		return 128 + sizeof(Mp4SeekCommand) + sizeof(seekStartTS) +sizeof(forceReopen) + Command::getSerializeSize();
	}

	uint64_t seekStartTS = 0;
	bool forceReopen = false;
private:

	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int)
	{
		ar& boost::serialization::base_object<Command>(*this);
		ar& seekStartTS;
		ar& forceReopen;
	}
};

//NVRCommands

class NVRCommandRecord : public Command
{
public:
	NVRCommandRecord() : Command(Command::CommandType::NVRCommandRecord)
	{
	}

	size_t getSerializeSize()
	{
		return Command::getSerializeSize() + sizeof(doRecording);
	}

	bool doRecording = false;

private:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int /* file_version */)
	{
		ar& boost::serialization::base_object<Command>(*this);
		ar& doRecording;
	}
};

class NVRCommandExport : public Command
{
public:
	NVRCommandExport() : Command(Command::CommandType::NVRCommandExport)
	{
	}

	size_t getSerializeSize()
	{
		return Command::getSerializeSize() + sizeof(startExportTS) + sizeof(stopExportTS);
	}

	uint64_t startExportTS = 0;
	uint64_t stopExportTS = 0;

private:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int /* file_version */)
	{
		ar& boost::serialization::base_object<Command>(*this);
		ar& startExportTS;
		ar& stopExportTS;
	}
};

class NVRCommandExportMMQ : public Command
{
public:
	NVRCommandExportMMQ() : Command(Command::CommandType::NVRCommandExportMMQ)
	{
	}

	size_t getSerializeSize()
	{
		return Command::getSerializeSize() + sizeof(startExportMMQ);
	}

	bool startExportMMQ = true;

private:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int /* file_version */)
	{
		ar& boost::serialization::base_object<Command>(*this);
		ar& startExportMMQ;
	}
};


class NVRCommandView : public Command
{
public:
	NVRCommandView() : Command(Command::CommandType::NVRCommandView)
	{
	}

	size_t getSerializeSize()
	{
		return Command::getSerializeSize() + sizeof(doView);
	}

	bool doView = false;

private:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int /* file_version */)
	{
		ar& boost::serialization::base_object<Command>(*this);
		ar& doView;
	}
};

class NVRGoLive : public Command
{
public:
	NVRGoLive() : Command(Command::CommandType::NVRGoLive)
	{
	}

	size_t getSerializeSize()
	{
		return Command::getSerializeSize();
	}

private:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int /* file_version */)
	{
		ar& boost::serialization::base_object<Command>(*this);
	}
};

class NVRCommandExportView : public Command
{
public:
	NVRCommandExportView() : Command(Command::CommandType::NVRCommandExportView)
	{
	}

	size_t getSerializeSize()
	{
		return Command::getSerializeSize() + sizeof(startViewTS) + sizeof(stopViewTS);
	}

	uint64_t startViewTS = 0;
	uint64_t stopViewTS = 0;

private:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int /* file_version */)
	{
		ar& boost::serialization::base_object<Command>(*this);
		ar& startViewTS;
		ar& stopViewTS;
	}
};

class MP4WriterLastTS : public Command
{
public:
	MP4WriterLastTS() : Command(Command::CommandType::MP4WriterLastTS)
	{
	}

	size_t getSerializeSize()
	{
		return Command::getSerializeSize() + sizeof(lastWrittenTimeStamp) + sizeof(moduleId);
	}

	uint64_t lastWrittenTimeStamp = 0;
	std::string moduleId;

private:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int /* file_version */)
	{
		ar& boost::serialization::base_object<Command>(*this);
		ar& lastWrittenTimeStamp;
		ar& moduleId;
	}
};

class MMQtimestamps : public Command
{
public:
	MMQtimestamps() : Command(Command::CommandType::MMQtimestamps)
	{
	}

	size_t getSerializeSize()
	{
		return Command::getSerializeSize() + sizeof(firstTimeStamp) + sizeof(lastTimeStamp) + sizeof(nvrExportStart) + sizeof(nvrExportStop) +sizeof(moduleId);
	}

	uint64_t firstTimeStamp = 0;
	uint64_t lastTimeStamp = 0;
	uint64_t nvrExportStart = 0;
	uint64_t nvrExportStop = 0;
	std::string moduleId;

private:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int /* file_version */)
	{
		ar& boost::serialization::base_object<Command>(*this);
		ar& firstTimeStamp;
		ar& lastTimeStamp;
		ar& nvrExportStart;
		ar& nvrExportStop;
		ar& moduleId;
	}
};

class Rendertimestamp : public Command
{
public:
	Rendertimestamp() : Command(Command::CommandType::Rendertimestamp)
	{
	}

	size_t getSerializeSize()
	{
		return Command::getSerializeSize() + sizeof(currentTimeStamp) +sizeof(moduleId);
	}

	uint64_t currentTimeStamp = 0;
	std::string moduleId;

private:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int /* file_version */)
	{
		ar& boost::serialization::base_object<Command>(*this);
		ar& currentTimeStamp;
		ar& moduleId;
	}
};



class PlayPauseCommand : public Command
{
public:
	PlayPauseCommand() : Command(CommandType::PlayPause)
	{
	}

	PlayPauseCommand(float _speed, bool _direction) : Command(CommandType::PlayPause)
	{

		if (_speed != 0 && _speed != 1)
		{
			LOG_ERROR << "Fractional speed is not yet supported.";
			throw AIPException(AIP_FATAL, "Fractional speed is not yet supported.");
		}
		speed = _speed;
		direction = _direction;
	}

	size_t getSerializeSize()
	{
		return sizeof(PlayPauseCommand) + sizeof(speed) + sizeof(direction) + Command::getSerializeSize();
	}

	// play speed of the module at any given fps
	float speed = 1;
	// fwd = 1, bwd = 0
	bool direction = 1;
private:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int)
	{
		ar& boost::serialization::base_object<Command>(*this);
		ar& speed;
		ar& direction;
	}
};