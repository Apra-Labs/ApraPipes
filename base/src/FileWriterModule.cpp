#include "stdafx.h"
#include "FileWriterModule.h"
#include "FrameMetadata.h"

#include "FileSequenceDriver.h"
#include "Frame.h"

FileWriterModule::FileWriterModule(FileWriterModuleProps _props)
	:Module(SINK, "FileWriterModule", _props)
{
	mDriver = boost::shared_ptr<FileSequenceDriver>(new FileSequenceDriver(_props.strFullFileNameWithPattern, _props.append));
	mDriver->notifyPlay(true);
}

FileWriterModule::~FileWriterModule() {}

bool FileWriterModule::validateInputPins()
{
	if (getNumberOfInputPins() != 1)
	{		
		return false;
	}

	// 1 and only 1 input expected
	// any frame type is allowed. so no validations
	return true;
}

bool FileWriterModule::init() {
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
bool FileWriterModule::term() {
	auto ret = mDriver->Disconnect();
	auto moduleRet = Module::term();

	return ret && moduleRet;
}
bool FileWriterModule::process(frame_container& frames) 
{
	auto frame = frames.begin()->second;
	size_t buffer_size = frame->size();

	try
	{
		if (!mDriver->Write(static_cast<uint8_t*>(frame->data()), buffer_size) && mDriver->IsConnected())
		{
			LOG_FATAL << "write failed<>" << frame->fIndex;
		}
	}
	catch (...)
	{
		LOG_FATAL << "unknown exception<>" << frame->fIndex;
	}
	
	return true;
}
