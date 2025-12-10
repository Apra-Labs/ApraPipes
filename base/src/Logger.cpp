#include "Logger.h"

boost::shared_ptr<Logger> Logger::instance;
std::mutex Logger::logger_mutex;

void Logger::initLogger(LoggerProps props)
{
	std::unique_lock<std::mutex> lock(logger_mutex);
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
	
	// start thread if we has some form of logging
	if (mProps.enableConsoleLog ||mProps.enableFileLog)
	{
		Logger& logger = *this;
		myThread = std::thread(std::ref(logger));
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
	std::cout << "Logger exiting..." <<std::endl;
	try{
		if (myThread.joinable())// It is a real thread
		{
			mQue.setWake();
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
	catch(const std::exception& e)
	{
		std::cout << "Exception while logger exit: " << e.what() <<std::endl;
	}
	catch(...)
	{
		std::cout << "Uknown exception while logger exit"<<std::endl;
	}
	
}

void Logger::setListener(void(*cb)(const std::string&))
{
	auto logger = Logger::getLogger();
	logger->_setListener(cb);
}

void Logger::_setListener(void(*cb)(const std::string&))
{
	mListener = cb;
}
bool Logger::setLogLevel(const std::string& sSeverity)
{
	boost::log::trivial::severity_level log_severity=boost::log::trivial::severity_level::debug;
	bool bRC=true;
    if(!boost::log::trivial::from_string(sSeverity.c_str(),sSeverity.length(), log_severity))
	{
		std::cout 
		<< "Valid severity level is one of [trace, debug, info, warning, error, fatal]. Provided " 
		<< sSeverity
		<< " defaulting to " << log_severity;
		bRC=false;
	}
	setLogLevel(log_severity);
	return bRC;
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
	try{
		while (mRunning || mQue.size())
		{
			std::string message = mQue.try_pop_external();
			if (!message.empty())
			{
				process(message);
			}
		}
	}
	catch(const std::exception& e)
	{
		std::cout << "!!Logger thread exiting for : " << e.what() <<std::endl;
		return false;
	}
	catch(...)
	{
		std::cout << "!!Logger thread exiting for uknown exception"<<std::endl;
		return false;
	}
	
	return true;
}

bool Logger::process(const std::string& message)
{		
	try{
		BOOST_LOG_SEV(lg, boost::log::trivial::info) << message;	

		if (mListener)
		{
			mListener(message);
		}
	}
	catch(const std::exception& e)
	{
		std::cout << "!!logging raised exception: " << e.what() << ", log message follows "<<std::endl;
		std::cout << message<<std::endl;
		return false;
	}
	catch(...)
	{
		std::cout << "!!logging raised unknown exception, log message follows "<<std::endl;
		std::cout << message<<std::endl;
		return false;
	}
	
	return true;
}

bool Logger::push(boost::log::trivial::severity_level level, std::ostringstream& stream)
{
	//AK:first push should always return true else code after << gets stubbed !!
	if(stream.tellp()==0) return true;  //avaoid unwanted copy

	//ignore log if severity is lower
	if (level < mProps.logLevel)
	{
		return false;
	}
	
	//ignore log if logging is not enabled

	if (!mProps.enableConsoleLog && !mProps.enableFileLog)
	{
		return false;
	}

	//push to queue only if the thread is running
	if(mRunning && myThread.get_id()!=std::thread::id())	//log thread is running
	{
		mQue.push(stream.str());	
	}
	else
	{
		//the thread is not running, let's log it on the caller thread at least
		process(stream.str()); 
	}

	return false;
}

std::ostringstream & Logger::pre(std::ostringstream& stream, boost::log::trivial::severity_level lvl)
{
	if (lvl >= mProps.logLevel)
	{
		//add TS and Sev into the log
		stream << boost::posix_time::microsec_clock::local_time() << " [" << to_string(lvl) << "] ";
	}
	return stream;
}

std::ostringstream & Logger::aipexceptionPre(std::ostringstream& stream, boost::log::trivial::severity_level lvl,int type)
{
	if (lvl >= mProps.logLevel)
	{
		//add TS and Sev into the log
		stream << boost::posix_time::microsec_clock::local_time() <<" [AIPException<" << type << ">] ";
	}
	return stream;
}