#include "stdafx.h"
#include <boost/filesystem.hpp>
#include "FileWriterModule.h"
#include "FrameMetadata.h"

#include "FileSequenceDriver.h"
#include "Frame.h"

FileWriterModule::FileWriterModule(FileWriterModuleProps _props)
	:Module(SINK, "FileWriterModule", _props)
{
	boost::filesystem::path p(_props.strFullFileNameWithPattern);
	boost::filesystem::path dirPath = p.parent_path();
	
	if (!boost::filesystem::exists(dirPath))
	{
		boost::filesystem::create_directories(dirPath);
	}
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

	framemetadata_sp metadata = getFirstInputMetadata();
	FrameMetadata::MemType memType = metadata->getMemType();
	if (memType != FrameMetadata::MemType::HOST)
	{
		LOG_INFO << "<" << getId() << ">::validateInputPins input memType is expected to be HOST. Actual<" << memType << ">";
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
		LOG_INFO << "Can not read directory";
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
			LOG_INFO << "write failed<>" << frame->fIndex;
		}
	}
	catch (...)
	{
		LOG_INFO << "unknown exception<>" << frame->fIndex;
	}
	
	return true;
}