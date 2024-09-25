#include <boost/test/unit_test.hpp>
#include <boost/foreach.hpp>
#include <boost/chrono.hpp>

#include "PipeLine.h"
#include "Module.h"
#include "SimpleControlModule.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include "FrameContainerQueue.h"
#include "AIPExceptions.h"
#include "ExternalSourceModule.h"
#include "ExternalSinkModule.h"

BOOST_AUTO_TEST_SUITE(simpleControlModule_tests)

class TestModuleSrcProps :public ModuleProps
{
public:
	TestModuleSrcProps() : ModuleProps()
	{
	}
};

class TestModuleSrc : public Module
{
public:
	TestModuleSrc(TestModuleSrcProps props = TestModuleSrcProps()) : Module(SOURCE, "TestModuleSrc", props)
	{
		addOutputPin();
	}

	std::string addOutputPin()
	{
		outMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
		return Module::addOutputPin(outMetadata);
	}

	bool produce()
	{
		auto outPin = getOutputPinIdByType(FrameMetadata::FrameType::GENERAL);
		auto frame = makeFrame(10);
		frames.insert(make_pair(outPin, frame));

		send(frames);
		return true;
	}

	bool validateInputPins()
	{
		return true;
	}

	bool validateOutputPins()
	{
		return true;
	}

	bool validateInputOutputPins()
	{
		return true;
	}

	frame_container frames;
	framemetadata_sp outMetadata;
};

class TestModuleTransformProps : public ModuleProps
{
public:
	TestModuleTransformProps() : ModuleProps()
	{}
	~TestModuleTransformProps()
	{}
};

class TestModuleTransform : public Module
{
public:
	TestModuleTransform(TestModuleTransformProps props) : Module(TRANSFORM, "TestTransform", props)
	{
		addOutputPin();
	}

	bool init()
	{
		Module::init();
		return true;
	}

	std::string addOutputPin()
	{
		outMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
		return Module::addOutputPin(outMetadata);
	}

protected:

	bool process(frame_container &frames)
	{
		auto outPin = getOutputPinIdByType(FrameMetadata::FrameType::GENERAL);
		auto frame = makeFrame(10, outPin);
		frames.insert(make_pair(outPin, frame));
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

	bool validateInputOutputPins()
	{
		return true;
	}
private:
	std::string outPin;
	framemetadata_sp outMetadata;
};

class TestSink : public Module
{
public:
	TestSink() : Module(SINK, "TestSink", ModuleProps())
	{

	}

	virtual ~TestSink() {}

protected:
	bool validateInputPins()
	{
		return true;
	}

	bool process(frame_container& frames)
	{
		return true;
	}
};

struct SimpleControlModuleTests
{
	SimpleControlModuleTests(bool enableHealthCallback = true, int intervalInSecs = 1)
	{
		LoggerProps loggerProps;
		loggerProps.logLevel = boost::log::trivial::severity_level::info;
		Logger::setLogLevel(boost::log::trivial::severity_level::info);
		Logger::initLogger(loggerProps);

		auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::GENERAL));
		sourceMod = boost::shared_ptr<TestModuleSrc>(new TestModuleSrc);
		//auto source_pin_1 = sourceMod->addOutputPin(metadata);

		/* set transform module health callbacks */
		TestModuleTransformProps props;
		props.logHealth = true;
		props.enableHealthCallBack = enableHealthCallback;
		props.healthUpdateIntervalInSec = intervalInSecs;
		transformMod1 = boost::shared_ptr<TestModuleTransform>(new TestModuleTransform(props));

		sinkMod = boost::shared_ptr<TestSink>(new TestSink);

		auto simpleCtrlProps = SimpleControlModuleProps();
		simpleCtrl = boost::shared_ptr<SimpleControlModule>(new SimpleControlModule(simpleCtrlProps));

		p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	}
	~SimpleControlModuleTests() {}

	void initModules()
	{
		BOOST_TEST(sourceMod->init());
		BOOST_TEST(transformMod1->init());
		BOOST_TEST(sinkMod->init());
	}

	void createPipeline()
	{
		sourceMod->setNext(transformMod1);
		transformMod1->setNext(sinkMod);
	}

	void enrollModules()
	{
		simpleCtrl->enrollModule("transform_test_module", transformMod1);
	}

	void attachModulesToPipeline()
	{
		p->appendModule(sourceMod);
		p->addControlModule(simpleCtrl);
	}

	void initPipeline()
	{
		p->init();
		simpleCtrl->init();
	}

	void runPipeline()
	{
		p->run_all_threaded();
	}

	void startPipeline()
	{
		createPipeline();
		enrollModules();

		attachModulesToPipeline();

		initPipeline();

		runPipeline();

		return;
	}

	bool termModules()
	{
		sourceMod->term();
		transformMod1->term();
		sinkMod->term();
		return true;
	}

	bool stopPipeline()
	{
		p->stop();
		p->term();
		p->wait_for_all();
		p.reset();
		return true;
	}

	void addControlModule()
	{
		p->addControlModule(simpleCtrl);
	}

	boost::shared_ptr<TestModuleSrc> sourceMod;
	boost::shared_ptr<TestModuleTransform> transformMod1;
	boost::shared_ptr<TestSink> sinkMod;
	boost::shared_ptr<SimpleControlModule> simpleCtrl;
	boost::shared_ptr<PipeLine> p;
};

void TestCallackExtention(const APHealthObject* healthObj, unsigned int eventId)
{
	auto moduleId = healthObj->getModuleId();
	BOOST_TEST(moduleId.find("TestTransform") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(simpleControlModule_healthCallback)
{
	SimpleControlModuleTests t;
	t.simpleCtrl->register_healthCallback_extention(TestCallackExtention);

	t.startPipeline();
	t.addControlModule();
	boost::this_thread::sleep_for(boost::chrono::milliseconds(5000));
	t.stopPipeline();
}

BOOST_AUTO_TEST_SUITE_END()