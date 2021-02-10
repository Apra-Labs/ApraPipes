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

// Fatal errors
#define AIP_FATAL 7811

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

private:
	std::string message;
};

#define AIPException(_type,_message) AIP_Exception(_type,__FILE__,__LINE__,_message)
