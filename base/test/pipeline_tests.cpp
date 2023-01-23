#include "PipeLine.h"
#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include "Logger.h"
#include "test_utils.h"
#include "FrameMetadata.h"
#include "Module.h"
#include "FrameContainerQueue.h"
#include "commondefs.h"
//#include "FrameMetadata.h"
//#include "ArrayMetadata.h"
#include "Frame.h"
//#include "AIPExceptions.h"
//#include "stdafx.h"
//#include <boost/test/unit_test.hpp>
//#include <boost/foreach.hpp>
//#include <boost/chrono.hpp>
//#include <assert.h>
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
	// sounceModule2
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

		boost::shared_ptr<FrameContainerQueue> getQue() { return Module::getQue(); }
		frame_sp makeFrame(size_t size, string &pinId) { return Module::makeFrame(size, pinId); }
		bool send(frame_container& frames) { return Module::send(frames); }
		

	protected:
		bool process() 
		{
			return false;
		}
		bool produce() override
		{
			auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
			size_t fSize = 521;
			std::string fPinId = getOutputPinIdByType(FrameMetadata::FrameType::GENERAL);
			auto frame = makeFrame(fSize, fPinId);
		
			frame_container frames;
			frames.insert(std::make_pair(fPinId, frame));

			send(frames);

			return true;
		}

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

		boost::shared_ptr<FrameContainerQueue> getQue() { return Module::getQue(); }
		frame_sp makeFrame(size_t size, string pinId) { return Module::makeFrame(size, pinId); }
		
		bool send(frame_container& frames) { return Module::send(frames); }
	protected:
		//bool process() {return false;}
		bool process(frame_container& frames)
		{
			auto frame = getFrameByType(frames, FrameMetadata::FrameType::GENERAL);
			
			std::string fPinId = getOutputPinIdByType(FrameMetadata::FrameType::GENERAL);
			frames.insert(std::make_pair(fPinId, frame));
			
			send(frames);
			return true;
		}
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

		boost::shared_ptr<FrameContainerQueue> getQue() { return Module::getQue(); }

		//bool send(frame_container& frames) { return Module::send(frames); }
	protected:
		//bool process() {return false;}
		bool process(frame_container& frames)
		{
			return true;
		}
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
	auto metadata1 = framemetadata_sp(new FrameMetadata(FrameMetadata::GENERAL));
	m1->addOutputPin(metadata1);
	auto m2 = boost::shared_ptr<CheckThread::TransformModule>(new CheckThread::TransformModule(CheckThread::TransformModuleProps()));
	m1->setNext(m2);
	auto metadata2 = framemetadata_sp(new FrameMetadata(FrameMetadata::GENERAL));
	m2->addOutputPin(metadata2);
	auto m3 = boost::shared_ptr<CheckThread::TransformModule>(new CheckThread::TransformModule(CheckThread::TransformModuleProps()));
	m2->setNext(m3);
	auto metadata3 = framemetadata_sp(new FrameMetadata(FrameMetadata::GENERAL));
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

BOOST_AUTO_TEST_CASE(flushallqueuesTest)
{
		CheckThread f;
		LoggerProps logprops;
		logprops.enableConsoleLog = true;
		logprops.enableFileLog = false;
		logprops.logLevel = boost::log::trivial::severity_level::info;
		Logger::initLogger(logprops);

		auto m1 = boost::shared_ptr<CheckThread::SourceModule>(new CheckThread::SourceModule(CheckThread::SourceModuleProps()));
		auto metadata1 = framemetadata_sp(new FrameMetadata(FrameMetadata::GENERAL));
		m1->addOutputPin(metadata1);
		auto m2 = boost::shared_ptr<CheckThread::TransformModule>(new CheckThread::TransformModule(CheckThread::TransformModuleProps()));
		m1->setNext(m2);
		auto metadata2 = framemetadata_sp(new FrameMetadata(FrameMetadata::GENERAL));
		m2->addOutputPin(metadata2);
		auto m3 = boost::shared_ptr<CheckThread::TransformModule>(new CheckThread::TransformModule(CheckThread::TransformModuleProps()));
		m2->setNext(m3);
		auto metadata3 = framemetadata_sp(new FrameMetadata(FrameMetadata::GENERAL));
		m3->addOutputPin(metadata3);
		auto m4 = boost::shared_ptr<CheckThread::SinkModule>(new CheckThread::SinkModule(CheckThread::SinkModuleProps()));
		m3->setNext(m4);

		auto m2Que = m2->getQue();
		auto m3Que = m3->getQue();
		auto m4Que = m4->getQue();

		PipeLine p("test");
		p.appendModule(m1);
		p.init();

	    m1->step();
		m1->step();
		m1->step();

		//boost::this_thread::sleep_for(boost::chrono::seconds(1));
		BOOST_TEST(m2Que->size() == 3);
		BOOST_TEST(m3Que->size() == 0);
		BOOST_TEST(m4Que->size() == 0);

		m2->step();	
		m2->step();
		BOOST_TEST(m2Que->size() == 1);
		BOOST_TEST(m3Que->size() == 2);
		BOOST_TEST(m4Que->size() == 0);

		m3->step();
		BOOST_TEST(m2Que->size() == 1);
		BOOST_TEST(m3Que->size() == 1);
		BOOST_TEST(m4Que->size() == 1); 

		p.flushAllQueues();
		BOOST_TEST(m2Que->size() == 0);
		BOOST_TEST(m3Que->size() == 0);
		BOOST_TEST(m4Que->size() == 0);

		p.stop();
		p.term();
}
BOOST_AUTO_TEST_SUITE_END()