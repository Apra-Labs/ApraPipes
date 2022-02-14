#pragma once

#include <string>
#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>
#include "ThreadSafeQue.h"
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/log/utility/setup/console.hpp>


/*
Module level control of logs is a todo
*/

class LoggerProps
{
public:
	LoggerProps()
	{
		enableConsoleLog = true;
		enableFileLog = false;
		fileLogPath = "";
		logLevel = boost::log::trivial::severity_level::error;
	}

	bool enableConsoleLog;
	bool enableFileLog;
	std::string fileLogPath;
	boost::log::trivial::severity_level logLevel;
	int fps;
};

/*
Todos:
Module Name should be printed
Per Module log level should be enabled
Separate loglevel for file and console log
*/

class Logger {
public:			
	static void initLogger(LoggerProps props);
	static Logger* getLogger();
	static void setLogLevel(boost::log::trivial::severity_level severity);
	static void setListener(void(*cb)(std::string&));

	virtual ~Logger();	
	
	void _setLogLevel(boost::log::trivial::severity_level severity);
	void setConsoleLog(bool enableLog);
	void setFileLog(bool enableLog);
	bool push(boost::log::trivial::severity_level level, std::ostringstream& stream);

	void _setListener(void(*cb)(std::string&));
	
	std::ostringstream& pre(std::ostringstream& stream, boost::log::trivial::severity_level lvl);
	std::ostringstream& aipexceptionPre(std::ostringstream& stream, boost::log::trivial::severity_level lvl,int type);
	
	void operator()(); //to support boost::thread
private:	
	Logger(LoggerProps props);
	void initBoostLogger(LoggerProps props);

	threadsafe_que<std::string> mQue;
	boost::thread myThread;	
	bool run();
	bool process(std::string& message);
	bool mRunning;
	bool logDisabled;
	LoggerProps mProps;	

	void(*mListener)(std::string&);

	static boost::shared_ptr<Logger> instance;
	boost::log::sources::severity_logger< boost::log::trivial::severity_level > lg;
	boost::shared_ptr< boost::log::sinks::synchronous_sink< boost::log::sinks::text_ostream_backend > > mConsoleSink;
	boost::shared_ptr< boost::log::sinks::synchronous_sink< boost::log::sinks::text_file_backend > > mFileSink;
};

#define A_LOG_SEV(severity) for(std::ostringstream stream; Logger::getLogger()->push(severity, stream);) Logger::getLogger()->pre(stream, severity)
#define A_LOG(severity) A_LOG_SEV(severity) << __FILE__ << ":" << __LINE__ << ":"

#define LOG_TRACE  A_LOG(boost::log::trivial::severity_level::trace)
#define LOG_DEBUG  A_LOG(boost::log::trivial::severity_level::debug)
#define LOG_INFO   A_LOG(boost::log::trivial::severity_level::info)
#define LOG_WARNING A_LOG(boost::log::trivial::severity_level::warning)
#define LOG_ERROR   A_LOG(boost::log::trivial::severity_level::error)
#define LOG_FATAL   A_LOG(boost::log::trivial::severity_level::fatal)