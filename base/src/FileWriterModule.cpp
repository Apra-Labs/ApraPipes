#include "stdafx.h"
#include <boost/filesystem.hpp>
#include "FileWriterModule.h"
#include "FrameMetadata.h"
#include <sys/time.h>
#include <ctime>
#include "FileSequenceDriver.h"
#include "Frame.h"

class FileWriterModule::FileWriterModuleGetCurrentStatus : public Command
{
public:
    FileWriterModuleGetCurrentStatus() : Command(static_cast<Command::CommandType>(Command::CommandType::GetPipStatus))
    {
    }
    size_t getSerializeSize()
    {
        return Command::getSerializeSize();
    }

private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive &ar, const unsigned int /* file_version */)
    {
        ar &boost::serialization::base_object<Command>(*this);
    }
};

class FileWriterModule::FileWriterModuleSetNumberOfFrame : public Command
{
public:
	int m_noOfFrames;
    FileWriterModuleSetNumberOfFrame(int noOfFrames) : Command(static_cast<Command::CommandType>(Command::CommandType::SetNoFrameSave)), m_noOfFrames(noOfFrames)
    {
    }
    size_t getSerializeSize()
    {
        return Command::getSerializeSize() + sizeof(m_noOfFrames);
    }

private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive &ar, const unsigned int /* file_version */)
    {
        ar &boost::serialization::base_object<Command>(*this);
		ar &m_noOfFrames;
    }
};

class FileWriterModule::FileWriterRestartPipeline : public Command
{
public:
	FileWriterRestartPipeline() : Command(static_cast<Command::CommandType>(Command::CommandType::ModuleRestart))
	{
	}

	size_t getSerializeSize()
	{
		return Command::getSerializeSize();
	}

private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive &ar, const unsigned int /* file_version */)
    {
        ar &boost::serialization::base_object<Command>(*this);
    }
};

FileWriterModule::FileWriterModule(FileWriterModuleProps _props)
	: Module(SINK, "FileWriterModule", _props), props(_props)
{
	boost::filesystem::path p(_props.strFullFileNameWithPattern);
	boost::filesystem::path dirPath = p.parent_path();

	if (!boost::filesystem::exists(dirPath))
	{
		boost::filesystem::create_directories(dirPath);
	} // create directory if it not exist
	mDriver = boost::shared_ptr<FileSequenceDriver>(new FileSequenceDriver(_props.strFullFileNameWithPattern, _props.append));
	mDriver->notifyPlay(true); // enable play flag
							   // isEnabled = _props.saveAsViewableImage;
}

FileWriterModule::~FileWriterModule() {}

bool FileWriterModule::validateInputPins()
{
	if (getNumberOfInputPins() != 1)
	{
		return false;
	}

	framemetadata_sp metadata = getFirstInputMetadata();
	FrameMetadata::MemType memType = metadata->getMemType();
	if (memType != FrameMetadata::MemType::HOST)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input memType is expected to be HOST. Actual<" << memType << ">";
		return false;
	}

	// any frame type is allowed. so no validations
	return true;
}

bool FileWriterModule::init()
{
	if (!Module::init())
	{
		return false;
	}

	if (!mDriver->Connect())
	{
		LOG_ERROR << "Can not read directory";
		return false;
	}
	return true;
}

bool FileWriterModule::term()
{
	auto ret = mDriver->Disconnect();
	auto moduleRet = Module::term();

	return ret && moduleRet;
}

bool FileWriterModule::process(frame_container &frames)
{
	LOG_ERROR << "coming inside process of filewriter";
	auto frame = frames.begin()->second;
	
	if (isFrameEmpty(frame))
	{
		LOG_DEBUG << "<============================== Frame is Empty ============================================>";
		return true;
	}

	

	if(mDriver->framesCaptured < mDriver->noOFFramesToCapture)
	{
		mDriver->framesCaptured++;
	}
	if(mDriver->framesCaptured >= mDriver->noOFFramesToCapture)
	{
		mDriver->isPlaying= false;
		LOG_DEBUG << "File Writer Status Will be Set to False But Value ";
	}
	LOG_DEBUG << "Current Status < " << mDriver->isPlaying;	
	


	if (props.saveViewableImage)
	{
		mDriver->Write1(frame);
		boost::this_thread::sleep_for(boost::chrono::microseconds(2000));
	}
	else
	{
		try
		{
			if (!mDriver->Write(const_cast<const uint8_t *>(static_cast<uint8_t *>(frame->data())),
								frame->size()) &&
				mDriver->IsConnected())
			{
				LOG_FATAL << "write failed<>" << frame->fIndex;
			}
			boost::this_thread::sleep_for(boost::chrono::microseconds(2000));
		}
		catch (...)
		{
			LOG_FATAL << "unknown exception<>" << frame->fIndex;
		}
	}

	return true;
}

bool FileWriterModule::handleCommand(Command::CommandType type, frame_sp &frame)
{
    if (type == Command::CommandType::GetPipStatus)
    {
        FileWriterModuleGetCurrentStatus cmd;
        getCommand(cmd, frame);
        mDriver->getCurrentStatus();
    }
    else if (type == Command::CommandType::SetNoFrameSave)
    {
        FileWriterModuleSetNumberOfFrame cmd(0);
        getCommand(cmd, frame);
		LOG_ERROR << "Command Frames Get Number Of Frames " << cmd.m_noOfFrames;
        mDriver->setNoOfFrame(cmd.m_noOfFrames);
    }
	else if (type == Command::CommandType::ModuleRestart)
    {
        FileWriterRestartPipeline cmd;
        getCommand(cmd, frame);
        mDriver->resetInfo();
    }
    else
    {
        return Module::handleCommand(type, frame);
    }
}

bool FileWriterModule::getPipelineStatus()
{
	return mDriver->getCurrentStatus();
    // FileWriterModuleGetCurrentStatus cmd;
    // return queueCommand(cmd);
}

bool FileWriterModule::setNumberOfFrame(int x)
{
    FileWriterModuleSetNumberOfFrame cmd(x);
    return queueCommand(cmd);
}

bool FileWriterModule::restartPipeline()
{
    FileWriterRestartPipeline cmd;
    return queueCommand(cmd);
}
  