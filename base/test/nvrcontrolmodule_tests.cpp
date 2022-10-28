#include <boost/test/unit_test.hpp>
#include "stdafx.h"
#include "PipeLine.h"
#include "Module.h"
#include "Utils.h"
#include "NVRControlModule.h"
#include <string>

BOOST_AUTO_TEST_SUITE(nvrcontrolmodule_tests)

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
		bool process() { return false; }
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
		bool process() { return false; }
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
		SinkModule(SinkModuleProps props) :Module(SINK, "mp4WritersinkModule", props) {};
	protected:
		bool process() { return false; }
		bool validateOutputPins()
		{
			return true;
		}
		bool validateInputPins()
		{
			return true;
		}
	};
};

BOOST_AUTO_TEST_CASE(basic)
{
	CheckThread f;

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
	auto mControl = boost::shared_ptr<NVRControlModule>(new NVRControlModule(NVRControlModuleProps()));

	PipeLine p("test");
	// add all source  modules
	p.appendModule(m1);
	// add control module if any
	p.addControlModule(mControl);
	// init
	p.init();
	// control init - do inside pipeline init
	mControl->init();
	
	p.run_all_threaded();
	mControl->record(true);
	// dont need step in run_all_threaded
	mControl->step();
	mControl->record(false);
	// dont need step in run_all_threaded
	mControl->step();
	boost::this_thread::sleep_for(boost::chrono::seconds(10));
	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_SUITE_END()