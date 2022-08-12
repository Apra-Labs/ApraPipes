#include "PipeLine.h"
#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include "Logger.h"
#include "test_utils.h"
#include "FrameMetadata.h"
#include "Module.h"
#include<iostream>
#include<fstream>


BOOST_AUTO_TEST_SUITE(pipeline_tests)

std::unordered_map<std::string, bool> checkName = {
	{"sourceModule_1", false},
	{"transformModule_2", false},
	{"transformModule_3", false},
	{"sinkModule_4", false}
};

void listenerForThreadName(const std::string& msg)
{
	if (msg.find("sourceModule_1")) { checkName["sourceModule_1"] = true; };
	if (msg.find("transformModule_2")) { checkName["transformModule_2"] = true; };
	if (msg.find("transformModule_3")) { checkName["transformModule_3"] = true; };
	if (msg.find("sinkModule_4")) { checkName["sinkModule_4"] = true; };
}

struct CheckThread {
	class SourceModuleProps : public ModuleProps
	{
	public:
		SourceModuleProps() : ModuleProps()
		{};
	};
	class TransformModuleProps : public ModuleProps
	{
	public:
		TransformModuleProps() : ModuleProps()
		{};
	};
	class SinkModuleProps : public ModuleProps
	{
	public:
		SinkModuleProps() : ModuleProps()
		{};
	};

	class SourceModule : public Module
	{
	public:
		SourceModule(SourceModuleProps props) : Module(SOURCE, "sourceModule", props)
		{
		};

	protected:
		bool process() {}
		bool validateOutputPins()
		{
			return true;
		}
		bool validateInputPins()
		{
			return true;
		}
	};
	class TransformModule : public Module
	{
	public:
		TransformModule(TransformModuleProps props) :Module(TRANSFORM, "transformModule", props) {};
	protected:
		bool process() {}
		bool validateOutputPins()
		{
			return true;
		}
		bool validateInputPins()
		{
			return true;
		}
	};
	class SinkModule : public Module
	{
	public:
		SinkModule(SinkModuleProps props) :Module(SINK, "sinkModule", props) {};
	protected:
		bool process() {}
		bool validateOutputPins()
		{
			return true;
		}
		bool validateInputPins()
		{
			return true;
		}
	};
	
	CheckThread() {};
	~CheckThread() 
	{
		checkName["sourceModule_1"] = false;
		checkName["transformModule_2"] = false;
		checkName["transformModule_3"] = false;
		checkName["sinkModule_4"] = false;
	};
};

BOOST_AUTO_TEST_CASE(checkThreadname)
{
	CheckThread f;
	LoggerProps logprops;
	logprops.enableConsoleLog = true;
	logprops.enableFileLog = false;
	logprops.logLevel = boost::log::trivial::severity_level::info;
	Logger::initLogger(logprops);

	Logger::setListener(listenerForThreadName);

	auto m1 = boost::shared_ptr<CheckThread::SourceModule>(new CheckThread::SourceModule(CheckThread::SourceModuleProps()));
	auto metadata1 = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	m1->addOutputPin(metadata1);
	auto m2 = boost::shared_ptr<CheckThread::TransformModule>(new CheckThread::TransformModule(CheckThread::TransformModuleProps()));
	m1->setNext(m2);
	auto metadata2 = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	m2->addOutputPin(metadata2);
	auto m3 = boost::shared_ptr<CheckThread::TransformModule>(new CheckThread::TransformModule(CheckThread::TransformModuleProps()));
	m2->setNext(m3);
	auto metadata3 = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	m3->addOutputPin(metadata3);
	auto m4 = boost::shared_ptr<CheckThread::SinkModule>(new CheckThread::SinkModule(CheckThread::SinkModuleProps()));
	m3->setNext(m4);

	PipeLine p("test");
	p.appendModule(m1);
	p.init();
	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(10));
	p.stop();
	p.term();
	p.wait_for_all();

	// stop getting the logs
	Logger::setListener(nullptr);
	for_each(checkName.begin(), checkName.end(), [](auto& checkNamePair) { BOOST_TEST((checkNamePair.second) == true);});
}
BOOST_AUTO_TEST_SUITE_END()