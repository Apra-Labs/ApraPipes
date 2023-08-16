#pragma once
#include "Frame.h"
#include "Utils.h"

class Mp4ErrorFrame : public Frame {
public:
	enum Mp4ErrorFrameType
	{
		MP4_SEEK,
		MP4_STEP
	};

	Mp4ErrorFrame() {}
	Mp4ErrorFrame(int _errorType, int _errorCode, std::string &_errorMsg)
	{
		errorType = _errorType;
		errorCode = _errorCode;
		errorMsg = _errorMsg;
	}

	Mp4ErrorFrame(int _errorType, int _errorCode, std::string &_errorMsg, int _openErrorCode, uint64_t& _errorMp4TS)
	{
		errorType = _errorType;
		errorCode = _errorCode;
		errorMsg = _errorMsg;
		openErrorCode = _openErrorCode;
		errorMp4TS = _errorMp4TS;
	}

	bool isMp4ErrorFrame()
	{
		return true;
	}

	int errorType; // SEEK/STEP
	int errorCode; // defined in AIPExceptions.h
	uint64_t errorMp4TS = 0; // skipTS in randomSeek, lastFrameTS in step
	int openErrorCode = 0; // expose some libmp4 error codes
	std::string errorMsg; // keep chars < 500
};