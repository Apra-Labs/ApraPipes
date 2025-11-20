#include "stdafx.h"
#include <filesystem>
#include "FileWriterModule.h"
#include "FrameMetadata.h"

#include "FileSequenceDriver.h"
#include "Frame.h"

FileWriterModule::FileWriterModule(FileWriterModuleProps _props)
	:Module(SINK, "FileWriterModule", _props)
{
	std::filesystem::path p(_props.strFullFileNameWithPattern);
	std::filesystem::path dirPath = p.parent_path();

	if (!std::filesystem::exists(dirPath))
	{
		std::filesystem::create_directories(dirPath);
	}
	mDriver = std::make_shared<FileSequenceDriver>(_props.strFullFileNameWithPattern, _props.append);
	mDriver->notifyPlay(true);
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
	try
	{
		if (!mDriver->Write(const_cast<const uint8_t *>(static_cast<uint8_t *>(frame->data())),
							frame->size()) &&
			mDriver->IsConnected())
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