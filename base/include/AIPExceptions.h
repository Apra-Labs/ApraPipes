#pragma once

#include <stdexcept>
#include "Logger.h"

#define AIP_CONNECTION_INCOMPATIBLE 7710
#define AIP_PIN_ATTRIBUTE_ERROR 7711
#define AIP_MODULE_INIT_ERROR 7712
#define AIP_ROI_OUTOFRANGE 7713
#define AIP_IMAGE_LOAD_FAILED 7714
#define AIP_PARAM_OUTOFRANGE 7715
#define AIP_UNIQUE_CONSTRAINT_FAILED 7716
#define AIP_PINS_VALIDATION_FAILED 7717
#define AIP_PIN_NOTFOUND 7718
#define AIP_NOTFOUND 7719
#define AIP_OUTOFORDER 7720
#define AIP_LENGTH_MISMATCH 7721
#define AIP_WRONG_STRUCTURE 7722
#define AIP_NOTIMPLEMENTED 7723
#define AIP_NOTSET 7725
#define AIP_NOTEXEPCTED 7726
#define MP4_OCOF_END 7726

// Fatal errors
#define AIP_FATAL 7811
#define MP4_NAME_PARSE_FAILED 7812
#define MP4_FILE_CLOSE_FAILED 7813
#define MP4_RELOAD_RESUME_FAILED 7814
#define MP4_OPEN_FILE_FAILED 7815
#define MP4_MISSING_VIDEOTRACK 7816
#define MP4_MISSING_START_TS 7817
#define MP4_TIME_RANGE_FETCH_FAILED 7818
#define MP4_SET_POINTER_END_FAILED 7819
#define MP4_SEEK_INSIDE_FILE_FAILED 7820
#define MP4_BUFFER_TOO_SMALL 7821
#define MP4_OCOF_EMPTY 7721
#define MP4_OCOF_MISSING_FILE 7822
#define MP4_OCOF_INVALID_DUR 7823
#define MP4_UNEXPECTED_STATE 7824
#define MODULE_ENROLLMENT_FAILED 7825
#define CTRL_MODULE_INVALID_STATE 7826


#define AIPException_LOG_SEV(severity,type) for(std::ostringstream stream; Logger::getLogger()->push(severity, stream);) Logger::getLogger()->aipexceptionPre(stream, severity,type)

// class referred from https://stackoverflow.com/a/8152888

class AIP_Exception : public std::runtime_error
{
public:                                                                                                                                                                                                                                                              
    /** Constructor (C++ STL strings).
     *  @param message The error message.
     */
    explicit AIP_Exception(int type,const std::string file,int line,const std::string logMessage) :
        runtime_error(std::to_string(type))
    {
        if (type > AIP_FATAL)
        {
            AIPException_LOG_SEV(boost::log::trivial::severity_level::fatal,type) << file << ":" << line << ":" << logMessage.c_str();
        }
        else
        {
            AIPException_LOG_SEV(boost::log::trivial::severity_level::error,type) << file << ":" << line << ":" << logMessage.c_str();
        }
        message = logMessage;
    }
    explicit AIP_Exception(int type,const std::string file,int line,const std::string logMessage, std::string _previosFile, std::string _nextFile) :
        runtime_error(std::to_string(type))
    {
        previousFile =  _previosFile;
        nextFile = _nextFile;
        AIPException_LOG_SEV(boost::log::trivial::severity_level::error,type) << file << ":" << line << ":" << previousFile.c_str() << ":" << nextFile.c_str() << ":" << logMessage.c_str();
    }
    /** Destructor.
     * Virtual to allow for subclassing.
     */
    virtual ~AIP_Exception() throw () {}
    int getCode()
    {
        return atoi(what());
    }
    std::string getError()
    {
        return message;
    }
    std::string getPreviousFile()
    {
        return previousFile;
    }
     std::string getNextFile()
    {
        return nextFile;
    }
private:
    std::string message;
    std::string previousFile;
    std::string nextFile;
};
class Mp4_Exception : public AIP_Exception
{
public:
    explicit Mp4_Exception(int type, const std::string file, int line, const std::string logMessage) :
        AIP_Exception(type, file, line, logMessage)
    {
    }
    explicit Mp4_Exception(int type, const std::string file, int line, int _openFileErrorCode, const std::string logMessage) :
        AIP_Exception(type, file, line, logMessage)
    {
        openFileErrorCode = _openFileErrorCode;
    }
    explicit Mp4_Exception(int type, const std::string file, int line, const std::string logMessage, std::string previosFile, std::string nextFile) :
        AIP_Exception(type, file, line, logMessage, previosFile, nextFile)
    {
    }
    int getOpenFileErrorCode()
    {
        return openFileErrorCode;
    }
private:
    int openFileErrorCode = 0;
};
#define AIPException(_type,_message) AIP_Exception(_type,__FILE__,__LINE__,_message)
#define Mp4Exception(_type,_message) Mp4_Exception(_type,__FILE__,__LINE__,_message)
#define Mp4ExceptionNoVideoTrack(_type,_message, _previosFile, _nextFile) Mp4_Exception(_type,__FILE__,__LINE__,_message,_previosFile,_nextFile)