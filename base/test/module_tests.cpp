#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include <boost/foreach.hpp>
#include <boost/chrono.hpp>

#include "PipeLine.h"
#include "Module.h"
#include "FrameMetadata.h"
#include "ArrayMetadata.h"
#include "Frame.h"
#include "FrameContainerQueue.h"
#include "AIPExceptions.h"

// NOTE: TESTS WHICH REQUIRE ANY ENVIRONMENT TO BE PRESENT BEFORE RUNNING ARE NOT UNIT TESTS !!!

BOOST_AUTO_TEST_SUITE(module_tests)

class TestModule : public Module
{
public:
	TestModule(Kind nature, string name, ModuleProps props) : Module(nature, name, props)
	{

	}

	virtual ~TestModule() {}

	size_t getNumberOfOutputPins(bool implicit = true) { return Module::getNumberOfOutputPins(implicit); }
	size_t getNumberOfInputPins() { return Module::getNumberOfInputPins(); }
	framemetadata_sp getFirstInputMetadata() { return Module::getFirstInputMetadata(); }
	framemetadata_sp getFirstOutputMetadata() { return Module::getFirstOutputMetadata(); }
	metadata_by_pin& getInputMetadata() { return Module::getInputMetadata(); }
	framefactory_by_pin& getOutputFrameFactory() { return Module::getOutputFrameFactory(); }
	framemetadata_sp getInputMetadataByType(int type) { return Module::getInputMetadataByType(type); }
	framemetadata_sp getOutputMetadataByType(int type) { return Module::getOutputMetadataByType(type); }
	int getNumberOfInputsByType(int type) { return Module::getNumberOfInputsByType(type); }
	int getNumberOfOutputsByType(int type) { return Module::getNumberOfOutputsByType(type); }	
	bool isMetadataEmpty(framemetadata_sp& metadata) { return Module::isMetadataEmpty(metadata); }
	bool isFrameEmpty(frame_sp& frame) { return Module::isFrameEmpty(frame); }
	string getInputPinIdByType(int type) { return Module::getInputPinIdByType(type); }
	string getOutputPinIdByType(int type) { return Module::getOutputPinIdByType(type); }

	void addInputPin(framemetadata_sp& metadata, string& pinId) { return Module::addInputPin(metadata, pinId); } // throws exception if validation fails
	Connections getConnections() { return Module::getConnections(); }

	boost_deque<frame_sp> getFrames(frame_container& frames) { return Module::getFrames(frames); }

	frame_sp makeFrame(size_t size, string pinId) { return Module::makeFrame(size, pinId); }
	frame_sp makeFrame(size_t size) { return Module::makeFrame(size); }
	
	bool send(frame_container& frames) { return Module::send(frames); }

	boost::shared_ptr<FrameContainerQueue> getQue() { return Module::getQue(); }
	frame_sp getFrameByType(frame_container& frames, int frameType) { return Module::getFrameByType(frames, frameType); }

	ModuleProps getProps() { return Module::getProps(); }
	void setProps(ModuleProps& props) { return Module::setProps(props); }
	void fillProps(ModuleProps& props) { return Module::fillProps(props); }

	bool processSourceQue() { return Module::processSourceQue(); }
	bool handlePausePlay(bool play) { return Module::handlePausePlay(play); }
	bool getPlayState() { return Module::getPlayState(); }
};

/*	
	protected
	virtual bool shouldTriggerSOS();
	
*/

struct FlushQTests
{
	FlushQTests() {}
	~FlushQTests() {}
	class TestModuleProps : public ModuleProps
	{
	public:
		TestModuleProps() :ModuleProps()
		{
		}
		~TestModuleProps()
		{}
	};
	class TestModule1 : public TestModule
	{
	public:
		TestModule1() : TestModule(SOURCE, "TestModule1", TestModuleProps())
		{

		}

		virtual ~TestModule1() {}

	protected:
		bool validateOutputPins() { return true; } // invoked with addOutputPin	};
	};
	class TestModule2 : public TestModule
	{
	public:
		TestModule2() : TestModule(TRANSFORM, "TestModule2", TestModuleProps())
		{

		}

		virtual ~TestModule2() {}

		bool process(frame_container& frames)
		{
			send(frames);
			return true;
		}
	protected:
		bool validateInputPins() { return true; } // invoked with setInputPin
		bool validateOutputPins() { return true; } // invoked with addOutputPin	
	};
	class TestModule3 : public TestModule
	{
	public:
		TestModule3() : TestModule(SINK, "TestModule3", TestModuleProps())
		{

		}
		virtual ~TestModule3() {}
		bool process(frame_container& frames)
		{
			LOG_INFO << "process TestModule3";
			return true;
		}
	protected:
		bool validateInputPins() { return true; }
	};
};

BOOST_AUTO_TEST_CASE(module_flushQ)
{
	auto m1 = boost::shared_ptr<TestModule>(new FlushQTests::TestModule1());
	auto metadata1 = framemetadata_sp(new FrameMetadata(FrameMetadata::GENERAL));
	auto pinId1 = string("s_p1");
	m1->addOutputPin(metadata1, pinId1);

	auto m2 = boost::shared_ptr<TestModule>(new FlushQTests::TestModule2());
	auto metadata2 = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
	auto pinId2 = string("s_p1");
	m2->addOutputPin(metadata2, pinId2);

	BOOST_TEST(m1->setNext(m2));
	BOOST_TEST(m1->init());
	BOOST_TEST(m2->init());

	auto genMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::GENERAL));
	auto frame = m1->makeFrame(512, "s_p1");
	frame_container frames;
	frames.insert(make_pair("s_p1", frame));
	m1->send(frames);
	m1->send(frames);
	
	auto m2Que = m2->getQue();
	BOOST_TEST(m2Que->size() == 2);

	m2->flushQue();
	BOOST_TEST(m2Que->size() == 0);
}

BOOST_AUTO_TEST_CASE(module_flushQRec_linear)
{
	auto m1 = boost::shared_ptr<TestModule>(new FlushQTests::TestModule1());
	auto metadata1 = framemetadata_sp(new FrameMetadata(FrameMetadata::GENERAL));
	auto pinId1 = string("s_p1");
	m1->addOutputPin(metadata1, pinId1);

	auto m2 = boost::shared_ptr<TestModule>(new FlushQTests::TestModule2());
	auto metadata2 = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
	auto pinId2 = string("s_p1");
	m2->addOutputPin(metadata2, pinId2);

	auto m3 = boost::shared_ptr<TestModule>(new FlushQTests::TestModule3());
	auto m4 = boost::shared_ptr<TestModule>(new FlushQTests::TestModule3());

	BOOST_TEST(m1->setNext(m2));
	BOOST_TEST(m1->setNext(m3));
	BOOST_TEST(m2->setNext(m4));
	BOOST_TEST(m1->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(m3->init());
	BOOST_TEST(m4->init());

	auto genMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::GENERAL));
	auto frame = m1->makeFrame(512, "s_p1");
	frame_container frames;
	frames.insert(make_pair("s_p1", frame));
	m1->send(frames);
	m1->send(frames);
	m1->send(frames);
	m1->send(frames);

	m2->step();
	m2->step();
	m3->step();
	m4->step();

	auto m2Que = m2->getQue();
	auto m3Que = m3->getQue();
	auto m4Que = m4->getQue();

	BOOST_TEST(m2Que->size() == 2);
	BOOST_TEST(m3Que->size() == 3);
	BOOST_TEST(m4Que->size() == 1);

	// flushQueRecursive for m2 - expectation is m2, m4 que become empty, rest remain same
	m2->flushQueRecursive();
	BOOST_TEST(m2Que->size() == 0);
	BOOST_TEST(m3Que->size() == 3);
	BOOST_TEST(m4Que->size() == 0);

	// m2 que should accept the next frame without requiring init
	m1->send(frames);
	BOOST_TEST(m2Que->size() == 1);
	BOOST_TEST(m3Que->size() == 4);
	BOOST_TEST(m4Que->size() == 0);

	// one more step
	m2->step();
	BOOST_TEST(m2Que->size() == 0);
	BOOST_TEST(m3Que->size() == 4);
	BOOST_TEST(m4Que->size() == 1);

	// flushQueRecursive for source - expectation - all queues are empty
	m1->flushQueRecursive();
	BOOST_TEST(m2Que->size() == 0);
	BOOST_TEST(m3Que->size() == 0);
	BOOST_TEST(m4Que->size() == 0);
}

BOOST_AUTO_TEST_CASE(module_flushQRec_branched)
{
	auto m1 = boost::shared_ptr<TestModule>(new FlushQTests::TestModule1());
	auto metadata1 = framemetadata_sp(new FrameMetadata(FrameMetadata::GENERAL));
	auto pinId1 = string("s_p1");
	m1->addOutputPin(metadata1, pinId1);

	auto m2 = boost::shared_ptr<TestModule>(new FlushQTests::TestModule2());
	auto metadata2 = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
	auto pinId2 = string("s_p1");
	m2->addOutputPin(metadata2, pinId2);

	auto m3 = boost::shared_ptr<TestModule>(new FlushQTests::TestModule3());
	auto m4 = boost::shared_ptr<TestModule>(new FlushQTests::TestModule3());

	auto m5 = boost::shared_ptr<TestModule>(new FlushQTests::TestModule2());
	m5->addOutputPin(metadata2, pinId2);

	auto m6 = boost::shared_ptr<TestModule>(new FlushQTests::TestModule3());
	auto m7 = boost::shared_ptr<TestModule>(new FlushQTests::TestModule3());

	BOOST_TEST(m1->setNext(m2));
	BOOST_TEST(m1->setNext(m3));
	BOOST_TEST(m1->setNext(m4));
	BOOST_TEST(m2->setNext(m5));
	BOOST_TEST(m2->setNext(m6));
	BOOST_TEST(m5->setNext(m7));
	BOOST_TEST(m1->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(m3->init());
	BOOST_TEST(m4->init());
	BOOST_TEST(m5->init());
	BOOST_TEST(m6->init());
	BOOST_TEST(m7->init());

	auto genMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::GENERAL));
	auto frame = m1->makeFrame(512, "s_p1");
	frame_container frames;
	frames.insert(make_pair("s_p1", frame));
	m1->send(frames);
	m1->send(frames);
	m1->send(frames);
	m1->send(frames);

	m2->step();
	m2->step();
	m2->step();
	m3->step();
	m4->step();
	m5->step();
	auto m2Que = m2->getQue();
	auto m3Que = m3->getQue();
	auto m4Que = m4->getQue();
	auto m5Que = m5->getQue();
	auto m6Que = m6->getQue();
	auto m7Que = m7->getQue();

	BOOST_TEST(m2Que->size() == 1);
	BOOST_TEST(m3Que->size() == 3);
	BOOST_TEST(m4Que->size() == 3);
	BOOST_TEST(m5Que->size() == 2);
	BOOST_TEST(m6Que->size() == 3);
	BOOST_TEST(m7Que->size() == 1);

	// flushQ on m5 - expectation is m5, m7 q become empty, rest remain same
	m5->flushQueRecursive();
	BOOST_TEST(m2Que->size() == 1);
	BOOST_TEST(m3Que->size() == 3);
	BOOST_TEST(m4Que->size() == 3);
	BOOST_TEST(m5Que->size() == 0);
	BOOST_TEST(m6Que->size() == 3);
	BOOST_TEST(m7Que->size() == 0);

	m1->send(frames);
	m1->send(frames);
	BOOST_TEST(m2Que->size() == 3);
	BOOST_TEST(m3Que->size() == 5);
	BOOST_TEST(m4Que->size() == 5);
	BOOST_TEST(m5Que->size() == 0);
	BOOST_TEST(m6Que->size() == 3);
	BOOST_TEST(m7Que->size() == 0);

	m2->step();
	m2->step();
	m5->step();
	BOOST_TEST(m2Que->size() == 1);
	BOOST_TEST(m3Que->size() == 5);
	BOOST_TEST(m4Que->size() == 5);
	BOOST_TEST(m5Que->size() == 1);
	BOOST_TEST(m6Que->size() == 5);
	BOOST_TEST(m7Que->size() == 1);

	// flushQueRecursive on m2 - expectation m2, m5, m6, m7 (subtree) que should be empty, rest remains same
	m2->flushQueRecursive();
	BOOST_TEST(m2Que->size() == 0);
	BOOST_TEST(m3Que->size() == 5);
	BOOST_TEST(m4Que->size() == 5);
	BOOST_TEST(m5Que->size() == 0);
	BOOST_TEST(m6Que->size() == 0);
	BOOST_TEST(m7Que->size() == 0);

	// flushQueRecursive for source - expectation all queues are empty
	m1->flushQueRecursive();
	BOOST_TEST(m2Que->size() == 0);
	BOOST_TEST(m3Que->size() == 0);
	BOOST_TEST(m4Que->size() == 0);
	BOOST_TEST(m5Que->size() == 0);
	BOOST_TEST(m6Que->size() == 0);
	BOOST_TEST(m7Que->size() == 0);
}

BOOST_AUTO_TEST_CASE(module_addOutputPin)
{
	class TestModule1 : public TestModule
	{
	public:
		TestModule1() : TestModule(SOURCE, "TestModule1", ModuleProps())
		{

		}

		virtual ~TestModule1() {}

	protected:
		bool validateOutputPins()
		{
			if (getNumberOfOutputPins() > 2)
			{
				return false;
			}

			pair<string, framefactory_sp> me; // map element	
			auto framefactoryByPin = getOutputFrameFactory();
			BOOST_FOREACH(me, framefactoryByPin) {
				FrameMetadata::FrameType frameType = me.second->getFrameMetadata()->getFrameType();
				if (frameType != FrameMetadata::RAW_IMAGE && frameType != FrameMetadata::ENCODED_IMAGE)
				{					
					return false;
				}
			}

			return true;
		}
	};


	auto m1 = boost::shared_ptr<TestModule>(new TestModule1());
	auto metadata1 = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	m1->addOutputPin(metadata1);

	auto outputMetadata = m1->getOutputFrameFactory();
	BOOST_TEST(outputMetadata.size() == 1);

	try
	{
		auto metadata2 = framemetadata_sp(new ArrayMetadata());
		m1->addOutputPin(metadata2);
		BOOST_TEST(false);
	}
	catch (AIP_Exception& exception)
	{
		BOOST_TEST(exception.getCode() == AIP_PINS_VALIDATION_FAILED);
		outputMetadata = m1->getOutputFrameFactory();
		BOOST_TEST(outputMetadata.size() == 1);
	}
	catch (...)
	{
		BOOST_TEST(false); // not expected to come here
	}

	auto metadata3 = framemetadata_sp(new RawImageMetadata());
	auto pinId = string("m1_p_1");
	m1->addOutputPin(metadata3, pinId);

	try 
	{
		auto metadata4 = framemetadata_sp(new RawImageMetadata());
		m1->addOutputPin(metadata4, pinId);
		BOOST_TEST(false);
	}
	catch (AIP_Exception& exception)
	{
		BOOST_TEST(exception.getCode() == AIP_UNIQUE_CONSTRAINT_FAILED);
		outputMetadata = m1->getOutputFrameFactory();
		BOOST_TEST(outputMetadata.size() == 2);
	}
	catch (...)
	{
		BOOST_TEST(false); // not expected to come here
	}

	outputMetadata = m1->getOutputFrameFactory();
	BOOST_TEST(outputMetadata.size() == 2);

	auto it = outputMetadata.cbegin();
	BOOST_TEST(it->second->getFrameMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);
	it++;
	BOOST_TEST(it->second->getFrameMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	BOOST_TEST(it->first == "m1_p_1");

	try
	{
		auto metadata5 = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
		m1->addOutputPin(metadata5);
		BOOST_TEST(false);
	}
	catch (AIP_Exception& exception)
	{
		BOOST_TEST(exception.getCode() == AIP_PINS_VALIDATION_FAILED);
		outputMetadata = m1->getOutputFrameFactory();
		BOOST_TEST(outputMetadata.size() == 2);
	}
	catch (...)
	{
		BOOST_TEST(false); // not expected to come here
	}
	
	BOOST_TEST(m1->getNumberOfOutputPins() == 2);
	BOOST_TEST(m1->getFirstOutputMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);
	BOOST_TEST(m1->getOutputMetadataByType(FrameMetadata::RAW_IMAGE)->getFrameType() == FrameMetadata::RAW_IMAGE);
	BOOST_TEST(m1->getOutputMetadataByType(FrameMetadata::ARRAY) == framemetadata_sp());
	BOOST_TEST(m1->getNumberOfOutputsByType(FrameMetadata::ENCODED_IMAGE) == 1);
	BOOST_TEST(m1->getNumberOfOutputsByType(FrameMetadata::ARRAY) == 0);
	auto metadata = m1->getOutputMetadataByType(FrameMetadata::ARRAY);
	BOOST_TEST(m1->isMetadataEmpty(metadata) == true);
	metadata = m1->getOutputMetadataByType(FrameMetadata::RAW_IMAGE);
	BOOST_TEST(m1->isMetadataEmpty(metadata) == false);
}

BOOST_AUTO_TEST_CASE(module_addInputPin)
{
	
	class TestModule1 : public TestModule
	{
	public:
		TestModule1() : TestModule(SINK, "TestModule1", ModuleProps())
		{

		}

		virtual ~TestModule1() {}

	protected:
		bool validateInputPins()
		{
			if (getNumberOfInputPins() > 2)
			{
				return false;
			}

			pair<string, framemetadata_sp> me; // map element	
			auto metadataByPin = getInputMetadata();
			BOOST_FOREACH(me, metadataByPin) {
				FrameMetadata::FrameType frameType = me.second->getFrameType();
				if (frameType != FrameMetadata::RAW_IMAGE && frameType != FrameMetadata::ENCODED_IMAGE)
				{
					return false;
				}
			}

			return true;
		}
	};

	auto m1 = boost::shared_ptr<TestModule>(new TestModule1());
	auto metadata1 = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	auto pinId1 = string("p1");
	m1->addInputPin(metadata1, pinId1);
	auto inputMetadata = m1->getInputMetadata();
	BOOST_TEST(inputMetadata.size() == 1);
	
	try
	{
		auto metadata2 = framemetadata_sp(new ArrayMetadata());
		auto pinId2 = string("p2");
		m1->addInputPin(metadata2, pinId2);
		BOOST_TEST(false);
	}
	catch (AIP_Exception& exception)
	{
		BOOST_TEST(exception.getCode() == AIP_PINS_VALIDATION_FAILED);
		inputMetadata = m1->getInputMetadata();
		BOOST_TEST(inputMetadata.size() == 1);
	}
	catch (...)
	{
		BOOST_TEST(false); // not expected to come here
	}
		
	auto metadata3 = framemetadata_sp(new RawImageMetadata());
	auto pinId = string("m1_p_1");
	m1->addInputPin(metadata3, pinId);
	inputMetadata = m1->getInputMetadata();
	BOOST_TEST(inputMetadata.size() == 2);

	try
	{
		auto metadata4 = framemetadata_sp(new RawImageMetadata());
		m1->addInputPin(metadata4, pinId);
		BOOST_TEST(false);
	}
	catch (AIP_Exception& exception)
	{
		BOOST_TEST(exception.getCode() == AIP_UNIQUE_CONSTRAINT_FAILED);
		inputMetadata = m1->getInputMetadata();
		BOOST_TEST(inputMetadata.size() == 2);
	}
	catch (...)
	{
		BOOST_TEST(false); // not expected to come here
	}
		
	auto it = inputMetadata.cbegin();
	BOOST_TEST(it->second->getFrameType() == FrameMetadata::RAW_IMAGE);
	it++;
	BOOST_TEST(it->second->getFrameType() == FrameMetadata::ENCODED_IMAGE);
	BOOST_TEST(it->first == "p1");

	try
	{
		auto metadata5 = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
		auto pinId3 = string("p3");
		m1->addInputPin(metadata5, pinId3);
		BOOST_TEST(false);
	}
	catch (AIP_Exception& exception)
	{
		BOOST_TEST(exception.getCode() == AIP_PINS_VALIDATION_FAILED);
		inputMetadata = m1->getInputMetadata();
		BOOST_TEST(inputMetadata.size() == 2);
	}
	catch (...)
	{
		BOOST_TEST(false); // not expected to come here
	}

	BOOST_TEST(m1->getNumberOfInputPins() == 2);
	BOOST_TEST(m1->getFirstInputMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	BOOST_TEST(m1->getInputMetadataByType(FrameMetadata::RAW_IMAGE)->getFrameType() == FrameMetadata::RAW_IMAGE);
	BOOST_TEST(m1->getInputMetadataByType(FrameMetadata::ARRAY) == framemetadata_sp());
	BOOST_TEST(m1->getNumberOfInputsByType(FrameMetadata::ENCODED_IMAGE) == 1);
	BOOST_TEST(m1->getNumberOfInputsByType(FrameMetadata::ARRAY) == 0);
	auto metadata = m1->getInputMetadataByType(FrameMetadata::ARRAY);
	BOOST_TEST(m1->isMetadataEmpty(metadata) == true);
	metadata = m1->getInputMetadataByType(FrameMetadata::RAW_IMAGE);
	BOOST_TEST(m1->isMetadataEmpty(metadata) == false);
}

BOOST_AUTO_TEST_CASE(module_setNext)
{

	class TestModule1 : public TestModule
	{
	public:
		TestModule1() : TestModule(SOURCE, "TestModule1", ModuleProps())
		{

		}

		virtual ~TestModule1() {}

	protected:
		bool validateOutputPins() { return true; } // invoked with addOutputPin	};
	};

	class TestModule2 : public TestModule
	{
	public:
		TestModule2() : TestModule(TRANSFORM, "TestModule2", ModuleProps())
		{

		}

		virtual ~TestModule2() {}

	protected:
		bool validateInputPins() { return true; } // invoked with setInputPin
		bool validateOutputPins() { return true; } // invoked with addOutputPin	
	};

	class TestModule3 : public TestModule
	{
	public:
		TestModule3() : TestModule(SINK, "TestModule3", ModuleProps())
		{

		}

		virtual ~TestModule3() {}

	protected:
		bool validateInputPins() { return false; } // invoked with setInputPin
	};

	auto m1 = boost::shared_ptr<TestModule>(new TestModule1());
	auto metadata1 = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	auto pinId1 = string("s_p1");
	m1->addOutputPin(metadata1, pinId1);
	auto metadata2 = framemetadata_sp(new RawImageMetadata());
	auto pinId2 = string("s_p2");
	m1->addOutputPin(metadata2, pinId2);

	auto m2 = boost::shared_ptr<TestModule>(new TestModule2());
	auto m3 = boost::shared_ptr<TestModule>(new TestModule2());
	auto m4 = boost::shared_ptr<TestModule>(new TestModule2());

	auto m5 = boost::shared_ptr<TestModule>(new TestModule3());

	vector<string> pinIdArr = { string("s_p1"), string("s_p2") };
	
	BOOST_TEST(m2->setNext(m1) == false); // next->getNature() < this->getNature()
	BOOST_TEST(m5->setNext(m1) == false); // next->getNature() < this->getNature()
	BOOST_TEST(m5->setNext(m2) == false); // next->getNature() < this->getNature()

	BOOST_TEST(m1->setNext(m2, pinIdArr));
	auto connectedModules = m1->getConnectedModules();
	BOOST_TEST(connectedModules.size() == 1);
	BOOST_TEST(connectedModules[0] == m2);
	auto connections = m1->getConnections();
	BOOST_TEST(connections.size() == 1);
	auto m2Id = m2->getId();
	BOOST_TEST( (connections.find(m2Id) != connections.end()) );
	BOOST_TEST(connections[m2Id].size() == 2);
	BOOST_TEST(connections[m2Id][0] == "s_p1");
	BOOST_TEST(connections[m2Id][1] == "s_p2");

	BOOST_TEST(m1->setNext(m2, pinIdArr) == false); // Connection already done

	pinIdArr = { "s_p2" };
	BOOST_TEST(m1->setNext(m3, pinIdArr));
	connectedModules = m1->getConnectedModules();
	BOOST_TEST(connectedModules.size() == 2);
	BOOST_TEST(connectedModules[1] == m3);
	connections = m1->getConnections();
	BOOST_TEST(connections.size() == 2);
	auto m3Id = m3->getId();
	BOOST_TEST((connections.find(m3Id) != connections.end()));
	BOOST_TEST(connections[m3Id].size() == 1);	
	BOOST_TEST(connections[m3Id][0] == "s_p2");
		
	BOOST_TEST(m1->setNext(m4));
	connectedModules = m1->getConnectedModules();
	BOOST_TEST(connectedModules.size() == 3);
	BOOST_TEST(connectedModules[2] == m4);
	connections = m1->getConnections();
	BOOST_TEST(connections.size() == 3);
	auto m4Id = m4->getId();
	BOOST_TEST((connections.find(m4Id) != connections.end()));
	BOOST_TEST(connections[m4Id].size() == 2);
	BOOST_TEST(connections[m4Id][0] == "s_p1");
	BOOST_TEST(connections[m4Id][1] == "s_p2");

	BOOST_TEST(m3->setNext(m4) == false); // No pins to connect
	BOOST_TEST(m3->setNext(m5) == false); // No pins to connect

	try
	{
		m1->setNext(m5);
		BOOST_TEST(false); // not expected to come here
	}
	catch (AIP_Exception& exception)
	{
		exception.getCode();
	}
	catch (...)
	{
		BOOST_TEST(false); // not expected to come here
	}

	connectedModules = m1->getConnectedModules();
	BOOST_TEST(connectedModules.size() == 3);
	connections = m1->getConnections();
	BOOST_TEST(connections.size() == 3);
}

BOOST_AUTO_TEST_CASE(disable_sieve)
{
	class TestModule1 : public TestModule
	{
	public:
		TestModule1() : TestModule(SOURCE, "TestModule1", ModuleProps())
		{

		}

		virtual ~TestModule1() {}

	protected:
		bool validateOutputPins() { return true; } // invoked with addOutputPin	};
	};

	class TestModule2 : public TestModule
	{
	public:
		TestModule2() : TestModule(TRANSFORM, "TestModule2", ModuleProps())
		{

		}

		virtual ~TestModule2() {}

	protected:
		bool validateInputPins() { return true; } // invoked with setInputPin
		bool validateOutputPins() { return true; } // invoked with addOutputPin	
	};

	class TestModule3 : public TestModule
	{
	public:
		TestModule3() : TestModule(TRANSFORM, "TestModule3", ModuleProps())
		{

		}

		virtual ~TestModule3() {}

	protected:
		bool validateInputPins() { return true; }
		bool validateOutputPins() { return true; }
	};

	class TestModule4 : public TestModule
	{
	public:
		TestModule4() : TestModule(TRANSFORM, "TestModule4", ModuleProps())
		{

		}

		virtual ~TestModule4() {}

	protected:
		bool validateInputPins() { return true; }
		bool validateOutputPins() { return true; }
	};

	class TestModule5 : public TestModule
	{
	public:
		TestModule5() : TestModule(SINK, "TestModule5", ModuleProps())
		{

		}
		virtual ~TestModule5() {}
	protected:
		bool validateInputPins() { return true; }
	};

	auto m1 = boost::shared_ptr<TestModule>(new TestModule1());
	auto metadata1 = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	auto pinId1 = string("s_p1");
	m1->addOutputPin(metadata1, pinId1);
	auto metadata2 = framemetadata_sp(new RawImageMetadata());
	auto pinId2 = string("s_p2");
	m1->addOutputPin(metadata2, pinId2);

	auto m2 = boost::shared_ptr<TestModule>(new TestModule2());
	auto metadata3 = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
	auto pinId3 = string("s_p3");
	m2->addOutputPin(metadata3, pinId3);

	auto m3 = boost::shared_ptr<TestModule>(new TestModule3());
	auto metadata4 = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::ARRAY));
	auto pinId4 = string("s_p4");
	m3->addOutputPin(metadata4, pinId4);

	vector<string> pinIdArr = { string("s_p1"), string("s_p2") };
	BOOST_TEST(m1->setNext(m2, pinIdArr));

	// m2->m3 with sieve = false - expected all input pins of m2 come to m3 as 
	BOOST_TEST(m2->setNext(m3, true, false));

	// input pin validation for m3
	BOOST_TEST(m3->getNumberOfInputPins() == 3);
	auto m3InputMetadata = m3->getInputMetadata();
	BOOST_TEST(m3InputMetadata["s_p1"]->getFrameType() == FrameMetadata::FrameType::ENCODED_IMAGE);
	BOOST_TEST(m3InputMetadata["s_p2"]->getFrameType() == FrameMetadata::FrameType::RAW_IMAGE);
	BOOST_TEST(m3InputMetadata["s_p3"]->getFrameType() == FrameMetadata::FrameType::GENERAL);


	// m3 -> m4 with sieve = true - expected only the output pin of m3 to be input of m4
	auto m4 = boost::shared_ptr<TestModule>(new TestModule4());
	m3->setNext(m4); // sieve = true by default
	BOOST_TEST(m4->getNumberOfInputPins() == 1);
	auto m4InputMetadata = m4->getInputMetadata();
	BOOST_TEST(m4InputMetadata["s_p4"]->getFrameType() == FrameMetadata::FrameType::ARRAY);

	// m3 -> m5 with sieve = false - expected all input pins + output pins of m3 to be input of m5
	auto m5 = boost::shared_ptr<TestModule>(new TestModule5());
	m3->setNext(m5, true, false);
	BOOST_TEST(m5->getNumberOfInputPins() == 4);
	auto m5InputMetadata = m5->getInputMetadata();
	BOOST_TEST(m5InputMetadata["s_p1"]->getFrameType() == FrameMetadata::FrameType::ENCODED_IMAGE);
	BOOST_TEST(m5InputMetadata["s_p2"]->getFrameType() == FrameMetadata::FrameType::RAW_IMAGE);
	BOOST_TEST(m5InputMetadata["s_p3"]->getFrameType() == FrameMetadata::FrameType::GENERAL);
	BOOST_TEST(m5InputMetadata["s_p4"]->getFrameType() == FrameMetadata::FrameType::ARRAY);
}

BOOST_AUTO_TEST_CASE(sieve_compl_outputPin_methods)
{
	class TestModule1 : public TestModule
	{
	public:
		TestModule1() : TestModule(SOURCE, "TestModule1", ModuleProps())
		{

		}

		virtual ~TestModule1() {}

	protected:
		bool validateOutputPins() { return true; } // invoked with addOutputPin	};
	};

	class TestModule2 : public TestModule
	{
	public:
		TestModule2() : TestModule(TRANSFORM, "TestModule2", ModuleProps())
		{

		}

		virtual ~TestModule2() {}

	protected:
		bool validateInputPins() { return true; } // invoked with setInputPin
		bool validateOutputPins() { return true; } // invoked with addOutputPin	
	};

	class TestModule3 : public TestModule
	{
	public:
		TestModule3() : TestModule(TRANSFORM, "TestModule3", ModuleProps())
		{

		}

		virtual ~TestModule3() {}

	protected:
		bool validateInputPins() { return true; }
		bool validateOutputPins() { return true; }
	};

	auto m1 = boost::shared_ptr<TestModule>(new TestModule1());
	auto metadata1 = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
	auto pinId1 = string("s_p1");
	m1->addOutputPin(metadata1, pinId1);
	auto metadata2 = framemetadata_sp(new RawImageMetadata());
	auto pinId2 = string("s_p2");
	m1->addOutputPin(metadata2, pinId2);
	BOOST_TEST(m1->getNumberOfOutputPins() == 2);

	auto m2 = boost::shared_ptr<TestModule>(new TestModule2());
	auto metadata3 = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
	auto pinId3 = string("s_p3");
	m2->addOutputPin(metadata3, pinId3);
	BOOST_TEST(m1->setNext(m2));
	BOOST_TEST(m2->getNumberOfOutputPins() == 1);

	auto m3 = boost::shared_ptr<TestModule>(new TestModule3());
	auto metadata4 = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::ARRAY));
	auto pinId4 = string("s_p4");
	m3->addOutputPin(metadata4, pinId4);
	BOOST_TEST(m2->setNext(m3, true, false));

	// with implicit behaviour 
	BOOST_TEST(m2->getNumberOfOutputPins() == 1);
	// with overridden implicit behaviour - no. of output pins should include i/p pins
	BOOST_TEST(m2->getNumberOfOutputPins(false) == 3);

	BOOST_TEST(m3->getNumberOfInputPins() == 3);
	BOOST_TEST(m3->getNumberOfOutputPins() == 1);
}

BOOST_AUTO_TEST_CASE(module_frame_container_utils)
{
	class TestModule2 : public TestModule
	{
	public:
		TestModule2(Kind nature, string name) : TestModule(nature, name, ModuleProps())
		{

		}

		virtual ~TestModule2() {}

	protected:
		bool validateInputPins() { return true; } // invoked with setInputPin
		bool validateOutputPins() { return true; } // invoked with addOutputPin	
	};
		
	auto m = boost::shared_ptr<TestModule>(new TestModule2(Module::SOURCE, "Module"));
	auto m1 = boost::shared_ptr<TestModule>(new TestModule2(Module::SINK, "Module"));
	auto m2 = boost::shared_ptr<TestModule>(new TestModule2(Module::SINK, "Module"));
	auto m3 = boost::shared_ptr<TestModule>(new TestModule2(Module::SINK, "Module"));
	auto m4 = boost::shared_ptr<TestModule>(new TestModule2(Module::SINK, "Module"));

	auto metadata1 = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE,FrameMetadata::MemType::HOST));
	auto metadata2 = framemetadata_sp(new FrameMetadata(FrameMetadata::H264_DATA,FrameMetadata::MemType::HOST));
	auto metadata3 = framemetadata_sp(new FrameMetadata(FrameMetadata::GENERAL,FrameMetadata::MemType::HOST));
	auto pin1Id = m->addOutputPin(metadata1);
	string pin2Id = "p2";
	m->addOutputPin(metadata2, pin2Id);
	auto pin3Id = m->addOutputPin(metadata3);

	m->setNext(m1);
	vector<string> pinIdArr = { pin1Id, pin3Id };
	m->setNext(m2, pinIdArr);
	vector<string> pinIdArr2 = { pin2Id };
	m->setNext(m3, pinIdArr2);

	BOOST_TEST(m->init());
	BOOST_TEST(m1->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(m3->init());
	BOOST_TEST(m4->init());
		
	auto frame1 = m->makeFrame(1023);
	BOOST_TEST((frame1->getMetadata() == metadata1));
	BOOST_TEST(((frame1->size() == 1023) && (frame1->data() != NULL)));
		
	auto frame2 = m->makeFrame(1025, pin2Id);
	BOOST_TEST((frame2->getMetadata() == metadata2));
	BOOST_TEST(((frame2->size() == 1025) && (frame2->data() != NULL)));

	auto frame3 = m->makeFrame(10000, pin3Id);
	BOOST_TEST((frame3->getMetadata() == metadata3));
	BOOST_TEST(((frame3->size() == 10000) && (frame3->data() != NULL)));
	
	frame_container frames_orig;
	frames_orig.insert(make_pair(pin1Id, frame1));
	frames_orig.insert(make_pair(pin2Id, frame2));
	frames_orig.insert(make_pair(pin3Id, frame3));

	BOOST_TEST(m->getFrameByType(frames_orig, FrameMetadata::ENCODED_IMAGE) == frame1);
	BOOST_TEST(m->getFrameByType(frames_orig, FrameMetadata::GENERAL) == frame3);
	BOOST_TEST(m->getFrameByType(frames_orig, FrameMetadata::AUDIO) == frame_sp());
	auto frame = m->getFrameByType(frames_orig, FrameMetadata::AUDIO);
	BOOST_TEST(m->isFrameEmpty(frame) == true);
	frame = m->getFrameByType(frames_orig, FrameMetadata::ENCODED_IMAGE);
	BOOST_TEST(m->isFrameEmpty(frame) == false);
	
	auto frames_arr = m->getFrames(frames_orig);
	BOOST_TEST(frames_arr.size() == 3);
	BOOST_TEST(frames_arr[0] == frame1); 
	BOOST_TEST(frames_arr[1] == frame3); // // because map is ordered by key automatically
	BOOST_TEST(frames_arr[2] == frame2);

	auto que = m1->getQue();
	auto frames = que->try_pop();
	BOOST_TEST(frames.size() == 0); // initially empty que
		
	m->send(frames_orig);
	que = m1->getQue();
	frames = que->try_pop();
	BOOST_TEST(frames.size() == 3);
	BOOST_TEST(frames[pin1Id] == frame1);
	BOOST_TEST(frames[pin2Id] == frame2);
	BOOST_TEST(frames[pin3Id] == frame3);
	frames = que->try_pop();
	BOOST_TEST(frames.size() == 0);

	que = m2->getQue();
	frames = que->try_pop();
	BOOST_TEST(frames.size() == 2);
	BOOST_TEST(frames[pin1Id] == frame1);
	BOOST_TEST(frames[pin3Id] == frame3);
	BOOST_TEST((frames.find(pin2Id) == frames.end()));
	frames = que->try_pop();
	BOOST_TEST(frames.size() == 0);

	que = m3->getQue();
	frames = que->try_pop();
	BOOST_TEST(frames.size() == 1);
	BOOST_TEST(frames[pin2Id] == frame2);
	frames = que->try_pop();
	BOOST_TEST(frames.size() == 0);

	que = m4->getQue();
	frames = que->try_pop();
	BOOST_TEST(frames.size() == 0);		
}

BOOST_AUTO_TEST_CASE(module_props)
{
	auto module = boost::shared_ptr<TestModule>(new TestModule(Module::SOURCE, "hola", ModuleProps(50, 40, true)));
	auto props = module->getProps();
	BOOST_TEST(props.fps == 50);
	BOOST_TEST(props.getQLen() == 40);
	BOOST_TEST(props.logHealth == true);

	ModuleProps newProps;
	module->fillProps(newProps);
	BOOST_TEST(props.fps == 50);
	BOOST_TEST(props.getQLen() == 40);
	BOOST_TEST(props.logHealth == true); 

	newProps.fps = 80;
	newProps.logHealth = false;
	module->setProps(newProps);

	props = module->getProps();
	BOOST_TEST(props.fps == 80);
	BOOST_TEST(props.getQLen() == 40);
	BOOST_TEST(props.logHealth == false);
}

void test_skip(boost::shared_ptr<TestModule>& source, boost::shared_ptr<TestModule>& transform1, boost::shared_ptr<TestModule>& transform2, boost::shared_ptr<TestModule>& transform3, boost::shared_ptr<FrameContainerQueue>& q1, boost::shared_ptr<FrameContainerQueue>& q2, boost::shared_ptr<FrameContainerQueue>& q3, boost::shared_ptr<FrameContainerQueue>& q4, int s1, int s2, int s3, int s4)
{
	source->step();

	transform1->step();
	BOOST_TEST(q1->try_pop().size() == s1);
	transform2->step();
	BOOST_TEST(q2->try_pop().size() == s2);
	transform3->step();
	BOOST_TEST(q3->try_pop().size() == s3);
	BOOST_TEST(q4->try_pop().size() == s4);
}

BOOST_AUTO_TEST_CASE(skip_test)
{
	// source -> transform1 -> sink1
	// source -> transform2 -> sink2
	// source -> transform3 -> sink3, sink4

	class DummyModule : public TestModule
	{
	public:
		DummyModule(Kind nature, string name) : TestModule(nature, name, ModuleProps())
		{

		}
		virtual ~DummyModule() {}

	protected:
		bool validateInputPins() { return true; } // invoked with setInputPin
		bool validateOutputPins() { return true; } // invoked with addOutputPin	

		bool produce()
		{
			auto pinId = getOutputFrameFactory().begin()->first;
			auto frame = makeFrame(1024, pinId);

			frame_container frames;
			frames.insert(std::make_pair(pinId, frame));

			send(frames);

			return true;
		}


		bool process(frame_container& frames)
		{			
			auto pinId = getOutputFrameFactory().begin()->first;
			auto frame = makeFrame(1024, pinId);

			frames.insert(std::make_pair(pinId, frame));

			send(frames);

			return true;
		}
	};

	auto source = boost::shared_ptr<TestModule>(new DummyModule(Module::SOURCE, "source"));
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
	source->addOutputPin(metadata);

	auto transform1 = boost::shared_ptr<TestModule>(new DummyModule(Module::TRANSFORM, "transform1"));
	source->setNext(transform1);
	auto metadata_t1 = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
	transform1->addOutputPin(metadata_t1);

	auto sink1 = boost::shared_ptr<TestModule>(new DummyModule(Module::SINK, "sink1"));
	transform1->setNext(sink1);	

	auto transform2 = boost::shared_ptr<TestModule>(new DummyModule(Module::TRANSFORM, "transform2"));
	source->setNext(transform2);
	auto metadata_t2 = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
	transform2->addOutputPin(metadata_t2);

	auto sink2 = boost::shared_ptr<TestModule>(new DummyModule(Module::SINK, "sink2"));
	transform2->setNext(sink2);

	auto transform3 = boost::shared_ptr<TestModule>(new DummyModule(Module::TRANSFORM, "transform3"));
	source->setNext(transform3);
	auto metadata_t3 = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
	transform3->addOutputPin(metadata_t3);

	auto sink3 = boost::shared_ptr<TestModule>(new DummyModule(Module::SINK, "sink3"));
	transform3->setNext(sink3);

	auto sink4 = boost::shared_ptr<TestModule>(new DummyModule(Module::SINK, "sink4"));
	transform3->setNext(sink4);

	BOOST_TEST(source->init());
	BOOST_TEST(transform1->init());
	BOOST_TEST(sink1->init());
	BOOST_TEST(transform2->init());
	BOOST_TEST(sink2->init());
	BOOST_TEST(transform3->init());
	BOOST_TEST(sink3->init());
	BOOST_TEST(sink4->init());

	auto q1 = sink1->getQue();
	auto q2 = sink2->getQue();
	auto q3 = sink3->getQue();
	auto q4 = sink4->getQue();

	auto props2 = transform2->getProps();
	auto props3 = transform3->getProps();

	for (auto i = 0; i < 5; i++)
	{
		// process everything by default
		test_skip(source, transform1, transform2, transform3, q1, q2, q3, q4, 1, 1, 1, 1);
	}

	
	// skip 1 out of 3
	props2.skipD = 3;
	props2.skipN = 1;
	transform2->setProps(props2);

	for (auto i = 0; i < 15; i++)
	{
		int s2 = 1;
		if ((i + 1) % 3 == 0)
		{
			s2 = 0;
		}
		test_skip(source, transform1, transform2, transform3, q1, q2, q3, q4, 1, s2, 1, 1);
	}

	props2.skipD = 1;
	props2.skipN = 0;
	transform2->setProps(props2);
	for (auto i = 0; i < 5; i++)
	{
		// process everything 
		test_skip(source, transform1, transform2, transform3, q1, q2, q3, q4, 1, 1, 1, 1);
	}

	// dont process anything
	props2.skipD = 1;
	props2.skipN = 1;
	transform2->setProps(props2);
	props3.skipD = 1;
	props3.skipN = 1;
	transform3->setProps(props3);
	for (auto i = 0; i < 5; i++)
	{
		// process everything 
		test_skip(source, transform1, transform2, transform3, q1, q2, q3, q4, 1, 0, 0, 0);
	}

	props2.skipD = 1;
	props2.skipN = 0;
	transform2->setProps(props2);
	props3.skipD = 3;
	props3.skipN = 2;
	transform3->setProps(props3);

	for (auto i = 0; i < 15; i++)
	{
		int s3 = 0;
		if ((i) % 3 == 0)
		{
			s3 = 1;
		}
		test_skip(source, transform1, transform2, transform3, q1, q2, q3, q4, 1, 1, s3, s3);
	}
}

BOOST_AUTO_TEST_CASE(stop)
{
	class TestModule2 : public TestModule
	{
	public:
		TestModule2(Kind nature, string name) : TestModule(nature, name, ModuleProps())
		{

		}

		virtual ~TestModule2() 
		{
			
		}

	protected:
		bool validateInputPins() { return true; } // invoked with setInputPin
		bool validateOutputPins() { return true; } // invoked with addOutputPin	
	};

	{
		auto m1 = boost::shared_ptr<TestModule2>(new TestModule2(Module::SOURCE, "hola1"));
		auto metadata1 = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
		m1->addOutputPin(metadata1);
		auto m2 = boost::shared_ptr<TestModule2>(new TestModule2(Module::TRANSFORM, "hola3"));
		m1->setNext(m2);
		auto metadata2 = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
		m2->addOutputPin(metadata2);
		auto m3 = boost::shared_ptr<TestModule2>(new TestModule2(Module::SINK, "hola2"));
		m2->setNext(m3);

		m1->init();
		m2->init();
		m3->init();

		m1->stop();
		m1->step();
		auto que = m2->getQue();
		auto frames = que->try_pop();
		BOOST_TEST(frames.size() == 0); // since modules are not running - stop is not propagated - nothing happens
	}

	{
		auto m1 = boost::shared_ptr<TestModule2>(new TestModule2(Module::SOURCE, "hola1"));
		auto metadata1 = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
		m1->addOutputPin(metadata1);
		auto m2 = boost::shared_ptr<TestModule2>(new TestModule2(Module::TRANSFORM, "hola3"));
		m1->setNext(m2);
		auto metadata2 = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
		m2->addOutputPin(metadata2);
		auto m3 = boost::shared_ptr<TestModule2>(new TestModule2(Module::SINK, "hola2"));
		m2->setNext(m3);

		m1->init();
		m2->init();
		m3->init();
		// running m1 in a thread
		auto t1 = std::thread(std::ref(*(m1.get())));
		std::this_thread::sleep_for(std::chrono::milliseconds(100));  // giving time to call step 
		m1->stop();
		m1->step();

		auto que = m2->getQue();
		auto frames = que->try_pop();
		BOOST_TEST(frames.size() == 1);
		BOOST_TEST(frames.begin()->second->isEoP() == true);
		que = m3->getQue();
		frames = que->try_pop();
		BOOST_TEST(frames.size() == 0); // since m2 is not running - stop is not propagated - nothing happens
		t1.join();
	}

	{
		auto m1 = boost::shared_ptr<TestModule2>(new TestModule2(Module::SOURCE, "hola1"));
		auto metadata1 = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
		m1->addOutputPin(metadata1);
		auto m2 = boost::shared_ptr<TestModule2>(new TestModule2(Module::TRANSFORM, "hola3"));
		m1->setNext(m2);
		auto metadata2 = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
		m2->addOutputPin(metadata2);
		auto m3 = boost::shared_ptr<TestModule2>(new TestModule2(Module::SINK, "hola2"));
		m2->setNext(m3);
		m1->init();
		m2->init();
		m3->init();
		//running m2 in a thread
		auto t2 = std::thread(std::ref(*(m2.get())));
		auto t1 = std::thread(std::ref(*(m1.get()))); // running m1 again - stop was called before so thread would have exited
		std::this_thread::sleep_for(std::chrono::milliseconds(100));  // giving time to call step
		m1->stop();
		m1->step();
		std::this_thread::sleep_for(std::chrono::milliseconds(100));  // giving time to call step 

		auto que = m3->getQue();
		auto frames = que->try_pop();
		BOOST_TEST(frames.size() == 1);
		BOOST_TEST(frames.begin()->second->isEoP() == true);

		t1.join();
		t2.join();
	}
}

BOOST_AUTO_TEST_CASE(stop_bug)
{
	

	class TestModule2 : public TestModule
	{
	public:
		TestModule2(Kind nature, string name, ModuleProps& props) : TestModule(nature, name, ModuleProps(props))
		{

		}

		virtual ~TestModule2() {}

	protected:
		bool validateInputPins() { return true; } // invoked with setInputPin
		bool validateOutputPins() { return true; } // invoked with addOutputPin	

		virtual bool process(frame_container& frames) { return true; }
		virtual bool produce() 
		{
			auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
			auto frame = makeFrame(1024, getOutputPinIdByType(metadata->getFrameType()));

			frame_container frames;
			frames.insert(std::make_pair(getOutputPinIdByType(metadata->getFrameType()), frame));

			send(frames);

			return true; 
		}
	};
	
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));

	ModuleProps props;
	props.qlen = 1;

	auto source1 = boost::shared_ptr<TestModule2>(new TestModule2(Module::SOURCE, "source", props));
	source1->addOutputPin(metadata);	
	auto source2 = boost::shared_ptr<TestModule2>(new TestModule2(Module::SOURCE, "source", props));
	source2->addOutputPin(metadata);
	auto sink = boost::shared_ptr<TestModule2>(new TestModule2(Module::SINK, "sink", props));
	source1->setNext(sink);
	source2->setNext(sink);

	source1->init();
	source2->init();
	sink->init();

	auto t1 = std::thread(std::ref(*(source1.get())));
	auto t2 = std::thread(std::ref(*(source2.get())));
	auto t3 = std::thread(std::ref(*(sink.get())));

	std::this_thread::sleep_for(std::chrono::milliseconds(100));  // giving time to call step

	source1->stop();
	std::this_thread::sleep_for(std::chrono::milliseconds(100));  // giving time

	source2->stop();
	// previously stuck in the above step
	std::this_thread::sleep_for(std::chrono::milliseconds(100));  // giving time
	
	t1.join();
	t2.join();
	t3.join();
}

BOOST_AUTO_TEST_CASE(pause_play_step, * boost::unit_test::disabled())
{
	class TestModule1 : public TestModule
	{
	public:
		TestModule1() : TestModule(SOURCE, "TestModule1", ModuleProps())
		{

		}

		virtual ~TestModule1() {}

	protected:
		bool produce()
		{
			return true;
		}

		bool validateOutputPins()
		{			
			return true;
		}
	};
	
	{
		// basic play
		auto m1 = boost::shared_ptr<TestModule1>(new TestModule1());
		auto metadata1 = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
		m1->addOutputPin(metadata1);

		m1->init();
		std::thread(std::ref(*(m1.get())));
		std::this_thread::sleep_for(std::chrono::milliseconds(100));  // giving time to call step

		BOOST_TEST(m1->getPlayState() == true);
		BOOST_TEST(m1->getTickCounter() != 0);

		m1->stop();
		std::this_thread::sleep_for(std::chrono::milliseconds(100));  // giving time to exit gracefully
	}

	{
		// play pause play
		auto m1 = boost::shared_ptr<TestModule1>(new TestModule1());
		auto metadata1 = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
		m1->addOutputPin(metadata1);

		m1->init();
		std::thread(std::ref(*(m1.get())));
		std::this_thread::sleep_for(std::chrono::milliseconds(100));  // giving time to call step

		BOOST_TEST(m1->getPlayState() == true);
		auto prevCounter = m1->getTickCounter();
		BOOST_TEST(prevCounter != 0);

		m1->play(false);
		std::this_thread::sleep_for(std::chrono::milliseconds(50));  
		prevCounter = m1->getTickCounter();

		std::this_thread::sleep_for(std::chrono::milliseconds(50));  
		BOOST_TEST(m1->getPlayState() == false);
		BOOST_TEST(prevCounter == m1->getTickCounter());

		m1->play(true);

		std::this_thread::sleep_for(std::chrono::milliseconds(50));  
		BOOST_TEST(m1->getPlayState() == true);
		BOOST_TEST(prevCounter != m1->getTickCounter());

		m1->stop();
		std::this_thread::sleep_for(std::chrono::milliseconds(100));  
	}

	{
		// pipeline play pause play

		auto m1 = boost::shared_ptr<TestModule1>(new TestModule1());
		auto metadata1 = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
		m1->addOutputPin(metadata1);

		PipeLine p("test");
		p.appendModule(m1);
		p.init();
		p.run_all_threaded();

		
		std::this_thread::sleep_for(std::chrono::milliseconds(100));  // giving time to call step 

		BOOST_TEST(m1->getPlayState() == true);
		auto prevCounter = m1->getTickCounter();
		BOOST_TEST(prevCounter != 0);

		p.pause();
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		prevCounter = m1->getTickCounter();

		std::this_thread::sleep_for(std::chrono::milliseconds(50)); 
		BOOST_TEST(m1->getPlayState() == false);
		BOOST_TEST(prevCounter == m1->getTickCounter());

		p.step();
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		BOOST_TEST(++prevCounter == m1->getTickCounter());		

		p.step();
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		BOOST_TEST(++prevCounter == m1->getTickCounter());

		p.play();

		std::this_thread::sleep_for(std::chrono::milliseconds(50)); 
		BOOST_TEST(m1->getPlayState() == true);
		BOOST_TEST(prevCounter != m1->getTickCounter());

		p.stop();
		p.term();

		p.wait_for_all();			
	}

	{
		// pipeline pause play pause

		auto m1 = boost::shared_ptr<TestModule1>(new TestModule1());
		auto metadata1 = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
		m1->addOutputPin(metadata1);

		PipeLine p("test");
		p.appendModule(m1);
		p.init();
		p.run_all_threaded_withpause();


		std::this_thread::sleep_for(std::chrono::milliseconds(100));  // giving time to call step 

		BOOST_TEST(m1->getPlayState() == false);
		BOOST_TEST(0 == m1->getTickCounter());

		p.play();

		std::this_thread::sleep_for(std::chrono::milliseconds(50)); 
		BOOST_TEST(m1->getPlayState() == true);
		auto prevCounter = m1->getTickCounter();
		BOOST_TEST(prevCounter != 0);
		
		p.pause();
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		prevCounter = m1->getTickCounter();

		std::this_thread::sleep_for(std::chrono::milliseconds(50)); 
		BOOST_TEST(m1->getPlayState() == false);
		BOOST_TEST(prevCounter == m1->getTickCounter());

		p.step();
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		BOOST_TEST(++prevCounter == m1->getTickCounter());

		p.step();
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		BOOST_TEST(++prevCounter == m1->getTickCounter());

		p.step();
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		BOOST_TEST(++prevCounter == m1->getTickCounter());

		p.stop();
		p.term();

		p.wait_for_all();
	}
}

BOOST_AUTO_TEST_CASE(relay)
{
	class TestModule2 : public TestModule
	{
	public:
		TestModule2(Kind nature, string name) : TestModule(nature, name, ModuleProps())
		{

		}

		virtual ~TestModule2() {}

	protected:
		bool validateInputPins() { return true; } // invoked with setInputPin
		bool validateOutputPins() { return true; } // invoked with addOutputPin	
	};

	auto source = boost::shared_ptr<TestModule2>(new TestModule2(Module::SOURCE, "source"));
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
	auto pinId = source->addOutputPin(metadata);
	auto sink1 = boost::shared_ptr<TestModule2>(new TestModule2(Module::SINK, "sink"));
	source->setNext(sink1, false);
	auto sink2 = boost::shared_ptr<TestModule2>(new TestModule2(Module::SINK, "sink"));
	source->setNext(sink2);

	source->init();
	sink1->init();
	sink2->init();

	auto sink1Que = sink1->getQue();
	auto sink2Que = sink2->getQue();

	auto frame = source->makeFrame(1023);
	frame_container frames_orig;
	frames_orig.insert(make_pair(pinId, frame));

	// 1 close 2 open
	source->send(frames_orig);
	BOOST_TEST(sink1Que->try_pop().size() == 0);
	BOOST_TEST(sink2Que->pop().size() == 1);

	// both open
	source->relay(sink1, true);
	source->step();
	source->send(frames_orig);
	BOOST_TEST(sink1Que->pop().size() == 1);
	BOOST_TEST(sink2Que->pop().size() == 1);
	
	// close 1
	source->relay(sink1, false);
	source->step();	
	source->send(frames_orig);
	BOOST_TEST(sink1Que->try_pop().size() == 0);
	BOOST_TEST(sink2Que->pop().size() == 1);

	// close 2
	source->relay(sink2, false);
	source->step();
	source->send(frames_orig);
	BOOST_TEST(sink1Que->try_pop().size() == 0);
	BOOST_TEST(sink2Que->try_pop().size() == 0);

	// open 1
	source->relay(sink1, true);
	source->step();
	source->send(frames_orig);
	BOOST_TEST(sink1Que->pop().size() == 1);
	BOOST_TEST(sink2Que->try_pop().size() == 0);

	// open 2
	source->relay(sink2, true);
	source->step();
	source->send(frames_orig);
	BOOST_TEST(sink1Que->pop().size() == 1);
	BOOST_TEST(sink2Que->pop().size() == 1);

}

BOOST_AUTO_TEST_CASE(pipeline_relay)
{
	class TestModule2 : public TestModule
	{
	public:
		TestModule2(Kind nature, string name) : TestModule(nature, name, ModuleProps())
		{

		}

		virtual ~TestModule2() {}

	protected:
		bool validateInputPins() { return true; } // invoked with setInputPin
		bool validateOutputPins() { return true; } // invoked with addOutputPin	
	};

	auto source = boost::shared_ptr<TestModule2>(new TestModule2(Module::SOURCE, "source"));
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
	auto pinId = source->addOutputPin(metadata);
	auto sink1 = boost::shared_ptr<TestModule2>(new TestModule2(Module::SINK, "sink"));
	source->setNext(sink1, false);
	auto sink2 = boost::shared_ptr<TestModule2>(new TestModule2(Module::SINK, "sink"));
	source->setNext(sink2);

	PipeLine p("test");
	p.appendModule(source);
	p.init();
	p.run_all_threaded();

	std::this_thread::sleep_for(std::chrono::milliseconds(100));  // giving time to call step 

	p.stop();
	p.term();
	p.wait_for_all();

	// if this test fails .. it never comes here and it will be stuck forever
}

BOOST_AUTO_TEST_CASE(fIndex2_propagate)
{
	class TestModule2 : public TestModule
	{
	public:
		TestModule2(Kind nature, string name) : TestModule(nature, name, ModuleProps())
		{

		}
		virtual ~TestModule2() {}

	protected:
		bool validateInputPins() { return true; } // invoked with setInputPin
		bool validateOutputPins() { return true; } // invoked with addOutputPin	

		bool process(frame_container& frames)
		{
			auto metadata1 = getOutputMetadataByType(FrameMetadata::ENCODED_IMAGE);
			auto metadata2 = getOutputMetadataByType(FrameMetadata::RAW_IMAGE);
			auto frame1 = makeFrame(1023, getOutputPinIdByType(FrameMetadata::ENCODED_IMAGE));
			auto frame2 = makeFrame(1024, getOutputPinIdByType(FrameMetadata::RAW_IMAGE));

			frames.insert(make_pair(getOutputPinIdByType(FrameMetadata::ENCODED_IMAGE), frame1));
			frames.insert(make_pair(getOutputPinIdByType(FrameMetadata::RAW_IMAGE), frame2));

			send(frames);
			return true;
		}
	};

	auto source = boost::shared_ptr<TestModule2>(new TestModule2(Module::SOURCE, "source"));
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
	auto pinId = source->addOutputPin(metadata);
	
	auto transform = boost::shared_ptr<TestModule2>(new TestModule2(Module::TRANSFORM, "transform"));
	source->setNext(transform);
	auto metadata2 = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::ENCODED_IMAGE));
	transform->addOutputPin(metadata2);
	auto metadata3 = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::RAW_IMAGE));
	transform->addOutputPin(metadata3);

	auto sink1 = boost::shared_ptr<TestModule2>(new TestModule2(Module::SINK, "sink"));
	transform->setNext(sink1);
	auto sink2 = boost::shared_ptr<TestModule2>(new TestModule2(Module::SINK, "sink"));
	transform->setNext(sink2);

	source->init();
	transform->init();
	sink1->init();
	sink2->init();

	auto sink1Que = sink1->getQue();
	auto sink2Que = sink2->getQue();

	auto frame = source->makeFrame(1023);
	frame->fIndex2 = 1011;
	frame_container frames_orig;
	frames_orig.insert(make_pair(pinId, frame));

	// 1 close 2 open
	source->send(frames_orig);
	transform->step();

	auto frames = sink1Que->pop();
	BOOST_TEST(frames.size() == 2);
	for (auto &it : frames)
	{
		BOOST_TEST(it.second->fIndex2 == 1011);
	}

	frames = sink2Que->pop();
	BOOST_TEST(frames.size() == 2);
	for (auto &it : frames)
	{
		BOOST_TEST(it.second->fIndex2 == 1011);
	}

	
}

BOOST_AUTO_TEST_CASE(feedbackmodule)
{
	class TestSource : public Module
	{
	public:
		TestSource() : Module(SOURCE, "TestSource", ModuleProps())
		{

		}

		virtual ~TestSource() {}

	protected:
		bool validateOutputPins()
		{			
			return true;
		}

		bool produce()
		{
			auto pinId = getOutputFrameFactory().begin()->first;
			auto frame = makeFrame(1024);

			frame_container frames;
			frames.insert(std::make_pair(pinId, frame));

			send(frames);

			return true;
		}
	};

	class TestTransform1Props: public ModuleProps
	{

	};

	class TestTransform1 : public Module
	{
	public:
		TestTransform1(TestTransform1Props props=TestTransform1Props()) : Module(TRANSFORM, "TestTransform1", ModuleProps(props))
		{

		}

		virtual ~TestTransform1() 
		{
			// LOG_ERROR << "DESTRUCTOR TRANSFORM1";
		}

		boost::shared_ptr<FrameContainerQueue> getQue() { return Module::getQue(); }

	protected:
		bool validateInputPins() { return true; } // invoked with setInputPin
		bool validateOutputPins() { return true; } // invoked with addOutputPin	

		bool process(frame_container& frames)
		{
			auto inputPinId = frames.begin()->first;
			//LOG_ERROR << inputPinId << " pin id";

			if (inputPinId.find("TestSource") == -1)
			{
				return true;
			}

			auto pinId = getOutputFrameFactory().begin()->first;
			auto frame = makeFrame(1024);

			frames.insert(std::make_pair(pinId, frame));

			send(frames);

			return true;
		}
	};

	class TestTransform2 : public Module
	{
	public:
		TestTransform2() : Module(TRANSFORM, "TestTransform2", ModuleProps())
		{

		}

		virtual ~TestTransform2() 
		{
			// LOG_ERROR << "DESTRUCTOR TRANSFORM2";
		}

		boost::shared_ptr<FrameContainerQueue> getQue() { return Module::getQue(); }

	protected:
		bool validateInputPins() { return true; } // invoked with setInputPin
		bool validateOutputPins() { return true; } // invoked with addOutputPin	

		bool process(frame_container& frames)
		{
			auto pinId = getOutputFrameFactory().begin()->first;
			auto frame = makeFrame(1024);

			frames.insert(std::make_pair(pinId, frame));

			send(frames);

			return true;
		}
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
		
	{		
		auto source = boost::shared_ptr<Module>(new TestSource());
		auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
		auto pinId = source->addOutputPin(metadata);
		
		auto transform1 = boost::shared_ptr<Module>(new TestTransform1());
		source->setNext(transform1);
		auto metadata2 = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
		transform1->addOutputPin(metadata2);
		
		auto transform2 = boost::shared_ptr<Module>(new TestTransform2());
		transform1->setNext(transform2);
		auto metadata3 = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
		transform2->addOutputPin(metadata3);
		
		transform2->setNext(transform1);
		
		auto sink = boost::shared_ptr<Module>(new TestSink());
		transform2->setNext(sink);
		
		PipeLine p("test");
		p.appendModule(source);

		BOOST_TEST(p.init() == false);
		BOOST_TEST(source->term());
		BOOST_TEST(transform1->term());
		BOOST_TEST(transform2->term());
		BOOST_TEST(sink->term());
	}

	{
		auto source = boost::shared_ptr<Module>(new TestSource());
		auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
		auto pinId = source->addOutputPin(metadata);

		auto transform1 = boost::shared_ptr<Module>(new TestTransform1());
		source->setNext(transform1);
		auto metadata2 = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
		transform1->addOutputPin(metadata2);

		auto transform2 = boost::shared_ptr<Module>(new TestTransform2());
		transform1->setNext(transform2);
		auto metadata3 = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
		transform2->addOutputPin(metadata3);

		transform2->addFeedback(transform1);

		auto sink = boost::shared_ptr<Module>(new TestSink());
		transform2->setNext(sink);

		PipeLine p("test");
		p.appendModule(source);

		BOOST_TEST(p.init());
		p.run_all_threaded();

		std::this_thread::sleep_for(std::chrono::seconds(1));

		p.stop();
		p.term();
		p.wait_for_all();
	}

	{
		auto source = boost::shared_ptr<Module>(new TestSource());
		auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
		auto pinId = source->addOutputPin(metadata);

		TestTransform1Props props;
		props.qlen = 1;
		props.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
		auto transform1 = boost::shared_ptr<TestTransform1>(new TestTransform1(props));
		source->setNext(transform1);
		auto metadata2 = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
		transform1->addOutputPin(metadata2);

		PipeLine p("test");
		p.appendModule(source);

		auto transform2 = boost::shared_ptr<TestTransform2>(new TestTransform2());
		transform1->setNext(transform2);
		auto metadata3 = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
		transform2->addOutputPin(metadata3);

		transform2->addFeedback(transform1);
		auto queue2 = transform2->getQue();

		BOOST_TEST(p.init());

		BOOST_TEST(transform2->init());
		p.run_all_threaded();

		std::this_thread::sleep_for(std::chrono::seconds(1));

		p.stop();
		p.term();
		std::this_thread::sleep_for(std::chrono::seconds(1));
		// source is stopped
		// transform 1 is stopped and waiting for transform2->push
		queue2->pop(); // free 1 slot in the que for the stop to be propagated
		p.wait_for_all();
		
		auto queue1 = transform1->getQue();
		BOOST_TEST(queue1->size() == 0);

		transform2->step(); // with this the queue will be full
		BOOST_TEST(queue1->size() == 0); // previously queue1 size was 1 here - so que full
		transform2->step(); // previously this step was in deadlock
		BOOST_TEST(queue1->size() == 0);
	}
	LOG_INFO << "COMPLETED";
}

BOOST_AUTO_TEST_SUITE_END()