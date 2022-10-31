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
		Mp4Seek
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

class Mp4SeekCommand : public Command
{
public:
	Mp4SeekCommand() : Command(CommandType::Mp4Seek)
	{

	}

	Mp4SeekCommand(uint64_t _skipTS) : Command(CommandType::Mp4Seek)
	{
		skipTS = _skipTS;
	}

	size_t getSerializeSize()
	{
		return 128 + sizeof(Mp4SeekCommand) + sizeof(skipTS) + Command::getSerializeSize();
	}

	uint64_t skipTS = 0;
private:

	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int)
	{
		ar& boost::serialization::base_object<Command>(*this);
		ar& skipTS;
	}
};