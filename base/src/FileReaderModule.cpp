#include "stdafx.h"
#include "FileReaderModule.h"
#include "FrameMetadata.h"
#include "FileSequenceDriver.h"
#include "Frame.h"
#include "Command.h"

FileReaderModule::FileReaderModule(FileReaderModuleProps _props)
	:Module(SOURCE, "FileReaderModule", _props), mProps(_props), mCache(false)
{
	mDriver = boost::shared_ptr<FileSequenceDriver>(new FileSequenceDriver(mProps.strFullFileNameWithPattern, mProps.startIndex, mProps.maxIndex, mProps.readLoop, mProps.files));
}

FileReaderModule::~FileReaderModule() {}

bool FileReaderModule::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{		
		return false;
	}

	// 1 and only 1 output expected
	// any frame type is allowed. so no validations
	return true;
}

bool FileReaderModule::init() {
	if (!Module::init())
	{
		return false;
	}

	if (!mDriver->Connect())
	{
		LOG_ERROR << "Can not read directory";
		return false;
	}

	mCache = mDriver->canCache();

	return true;
}
bool FileReaderModule::term() {
	auto ret = mDriver->Disconnect();
	auto moduleRet = Module::term();

	return ret && moduleRet;
}
bool FileReaderModule::produce() 
{	
	auto metadata = getOutputMetadata().begin()->second;
	if (mCache && mFrames.size() == 1)
	{
		// used for performance tests
		frame_container frames;
		auto& cachedFrame = mFrames.begin()->second; 
		auto frame = frame_sp(new Frame(cachedFrame->data(), cachedFrame->size(), boost::shared_ptr<FrameFactory>()));
		frame->setMetadata(metadata);
		frames.insert(make_pair(getOutputMetadata().begin()->first, frame));
		send(frames);
		return true;
	}
	
	
	auto buffer = makeBuffer(mProps.maxFileSize, metadata->getMemType());
	size_t buffer_size = mProps.maxFileSize;
	uint64_t fIndex2 = 0;
	if (!mDriver->ReadP(static_cast<uint8_t*>(buffer->data()), buffer_size, fIndex2))
	{
		if (buffer_size > mProps.maxFileSize)
		{
			mProps.maxFileSize = buffer_size;
			return produce(); //danger: may be reentrant
		}
		return false;
	}
	
	auto frame = makeFrame(buffer, buffer_size, metadata);
	frame->fIndex2 = fIndex2;
	
	frame_container frames;
	frames.insert(make_pair(getOutputMetadata().begin()->first, frame));
	send(frames);

	if (mCache)
	{
		mFrames = frames;
	}

	return true;
}

void FileReaderModule::notifyPlay(bool play)
{
	mDriver->notifyPlay(play);
}

bool FileReaderModule::jump(uint64_t index)
{
	FileReaderModuleCommand cmd(index);
	
	return queueCommand(cmd);
}

bool FileReaderModule::handleCommand(Command::CommandType type, frame_sp& frame)
{
	if (type != Command::CommandType::FileReaderModule)
	{		
		return Module::handleCommand(type, frame); 
	}

	FileReaderModuleCommand cmd;
	getCommand(cmd, frame);

	mDriver->jump(cmd.getCurrentIndex());

	return true;
}

void FileReaderModule::setProps(FileReaderModuleProps& props)
{
	Module::addPropsToQueue(props);
}

FileReaderModuleProps FileReaderModule::getProps()
{
	fillProps(mProps);
	return mProps;
}

bool FileReaderModule::handlePropsChange(frame_sp& frame)
{	
	bool ret = Module::handlePropsChange(frame, mProps);

	mDriver->Disconnect();
	mDriver = boost::shared_ptr<FileSequenceDriver>(new FileSequenceDriver(mProps.strFullFileNameWithPattern, mProps.startIndex, mProps.maxIndex, mProps.readLoop, mProps.files));
	if (!mDriver->Connect())
	{
		LOG_ERROR << "Driver Connect Failed";
		return false;
	}

	mDriver->notifyPlay(getPlayState());

	sendEOS();

	return ret;
}