#include "Logger.h"

boost::shared_ptr<Logger> Logger::instance;

void Logger::initLogger(LoggerProps props)
{
	if (instance.get())
	{
		return;
	}

	instance.reset(new Logger(props));
}

Logger* Logger::getLogger()
{		
	if (instance.get())
	{
		return instance.get();
	}

	initLogger(LoggerProps());
	return instance.get();
}

Logger::Logger(LoggerProps props)
{
	mListener = nullptr;
	mProps = props;
	initBoostLogger(props);
	
	logDisabled = !mProps.enableConsoleLog && !mProps.enableFileLog;
	// start thread
	if (!logDisabled)
	{
		Logger& logger = *this;
		myThread = boost::thread(std::ref(logger));
	}
}

void Logger::initBoostLogger(LoggerProps props)
{
	if (props.enableConsoleLog)
	{
		mConsoleSink = boost::log::add_console_log();
	}

	if (!props.enableFileLog)
	{
		return;
	}

	if (props.fileLogPath.empty())
	{
		props.enableFileLog = false;
		return;
	}

	size_t index = props.fileLogPath.find_last_of(".");
	if (index == std::string::npos)
	{
		throw "invalid log file path. c:/users/developer/temp/abc.log";
	}
	props.fileLogPath = props.fileLogPath.substr(0, index) + "_%5N" + props.fileLogPath.substr(index);

	mFileSink = boost::log::add_file_log
	(
		boost::log::keywords::file_name = props.fileLogPath,                                        /*< file name pattern >*/
		boost::log::keywords::rotation_size = 10*1024*1024,                                   /*< rotate files every 10 MiB... >*/
		boost::log::keywords::time_based_rotation = boost::log::sinks::file::rotation_at_time_point(0, 0, 0), /*< ...or at midnight >*/		
		boost::log::keywords::auto_flush = true,
		boost::log::keywords::min_free_space = 25 * 1024 * 1024, /* 25 MB */
		boost::log::keywords::max_files = 25,
		boost::log::keywords::target = boost::filesystem::path(props.fileLogPath).parent_path().string()
	);
	
	boost::log::add_common_attributes();
}

Logger::~Logger()
{
	mRunning = false;
	
	if (!logDisabled)
	{
		mQue.setWake();
		myThread.~thread(); // not the ideal way - thread exits before processing the queue
		myThread.join();
	}

	if (mConsoleSink.get())
	{
		mConsoleSink->flush();
	}
	if (mFileSink.get())
	{
		mFileSink->flush();
	}

	mQue.clear();
}

void Logger::setListener(void(*cb)(std::string&))
{
	auto logger = Logger::getLogger();
	logger->_setListener(cb);
}

void Logger::_setListener(void(*cb)(std::string&))
{
	mListener = cb;
}

void Logger::setLogLevel(boost::log::trivial::severity_level severity)
{
	auto logger = Logger::getLogger();
	logger->_setLogLevel(severity);
}

void Logger::_setLogLevel(boost::log::trivial::severity_level severity)
{	
	mProps.logLevel = severity;
}

void Logger::setConsoleLog(bool enableLog)
{
	mProps.enableConsoleLog = enableLog;
}

void Logger::setFileLog(bool enableLog)
{
	mProps.enableFileLog = enableLog;
	if (mProps.fileLogPath.empty())
	{
		mProps.enableFileLog = false;
	}
}

void Logger::operator()() 
{
	run();
}

bool Logger::run()
{
	mRunning = true;
	while (mRunning || mQue.size())
	{
		std::string message = mQue.try_pop_external();
		if (!message.empty())
		{
			process(message);
		}
	}
		
	return true;
}

bool Logger::process(std::string& message)
{		
	BOOST_LOG_SEV(lg, boost::log::trivial::info) << message;	

	if (mListener)
	{
		mListener(message);
	}
	
	return true;
}

bool Logger::push(boost::log::trivial::severity_level level, std::ostringstream& stream)
{
	if (logDisabled)
	{
		return false;
	}

	if (stream.str().empty())
	{
		return true;
	}

	if (level < mProps.logLevel)
	{
		return false;
	}
		
	mQue.push(stream.str());	

	return false;
}

std::ostringstream & Logger::pre(std::ostringstream& stream, boost::log::trivial::severity_level lvl)
{
	if (lvl >= mProps.logLevel)
	{
		//add TS and Sev into the log
		stream << boost::posix_time::microsec_clock::universal_time() << " [" << to_string(lvl) << "] ";
	}
	return stream;
}

std::ostringstream & Logger::aipexceptionPre(std::ostringstream& stream, boost::log::trivial::severity_level lvl,int type)
{
	if (lvl >= mProps.logLevel)
	{
		//add TS and Sev into the log
		stream << boost::posix_time::microsec_clock::universal_time() <<" [AIPException<" << type << ">] ";
	}
	return stream;
}