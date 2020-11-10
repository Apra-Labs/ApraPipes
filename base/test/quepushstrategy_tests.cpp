#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include "FrameContainerQueue.h"
#include "QuePushStrategy.h"

// NOTE: TESTS WHICH REQUIRE ANY ENVIRONMENT TO BE PRESENT BEFORE RUNNING ARE NOT UNIT TESTS !!!


BOOST_AUTO_TEST_SUITE(quepushstrategy_tests)


BOOST_AUTO_TEST_CASE(blocking)
{
	boost::shared_ptr<FrameContainerQueue> que1 = boost::shared_ptr<FrameContainerQueue>(new FrameContainerQueue(10));
	boost::shared_ptr<FrameContainerQueue> que2 = boost::shared_ptr<FrameContainerQueue>(new FrameContainerQueue(2));

	auto moduleId = std::string("HOLA");
	auto strategy = QuePushStrategy::getStrategy(QuePushStrategy::BLOCKING, moduleId);
	std::string pin1 = "a";
	std::string pin2 = "b";

	strategy->addQue(pin1, que1);
	strategy->addQue(pin2, que2);

	frame_container frames;

	strategy->push(pin1, frames);
	strategy->push(pin2, frames);
	strategy->flush();

	BOOST_TEST(que1->size() == 1);
	BOOST_TEST(que2->size() == 1);

}

BOOST_AUTO_TEST_CASE(NON_BLOCKING_ANY)
{
	boost::shared_ptr<FrameContainerQueue> que1 = boost::shared_ptr<FrameContainerQueue>(new FrameContainerQueue(10));
	boost::shared_ptr<FrameContainerQueue> que2 = boost::shared_ptr<FrameContainerQueue>(new FrameContainerQueue(2));

	auto moduleId = std::string("HOLA");
	auto strategy = QuePushStrategy::getStrategy(QuePushStrategy::NON_BLOCKING_ANY, moduleId);
	std::string pin1 = "a";
	std::string pin2 = "b";

	strategy->addQue(pin1, que1);
	strategy->addQue(pin2, que2);

	frame_container frames;

	strategy->push(pin1, frames);
	strategy->push(pin2, frames);
	strategy->flush();

	BOOST_TEST(que1->size() == 1);
	BOOST_TEST(que2->size() == 1);


	strategy->push(pin1, frames);
	strategy->push(pin2, frames);
	strategy->flush();

	BOOST_TEST(que1->size() == 2);
	BOOST_TEST(que2->size() == 2);

	strategy->push(pin1, frames);
	strategy->push(pin2, frames);
	strategy->flush();

	// because que2 has max capacity of 2 ..q2 fails but q1 success
	BOOST_TEST(que1->size() == 3);
	BOOST_TEST(que2->size() == 2);

	strategy->push(pin1, frames);
	strategy->flush();
		
	BOOST_TEST(que1->size() == 4);
	BOOST_TEST(que2->size() == 2);

	que2->pop();
	BOOST_TEST(que2->size() == 1);

	strategy->push(pin1, frames);
	strategy->push(pin2, frames);
	strategy->flush();

	BOOST_TEST(que1->size() == 5);
	BOOST_TEST(que2->size() == 2);
}

BOOST_AUTO_TEST_CASE(NonBlockingAllOrNonePushStrategy)
{
	boost::shared_ptr<FrameContainerQueue> que1 = boost::shared_ptr<FrameContainerQueue>(new FrameContainerQueue(10));
	boost::shared_ptr<FrameContainerQueue> que2 = boost::shared_ptr<FrameContainerQueue>(new FrameContainerQueue(2));

	auto moduleId = std::string("HOLA");
	auto strategy = QuePushStrategy::getStrategy(QuePushStrategy::NON_BLOCKING_ALL_OR_NONE, moduleId);
	std::string pin1 = "a";
	std::string pin2 = "b";

	strategy->addQue(pin1, que1);
	strategy->addQue(pin2, que2);

	frame_container frames;

	strategy->push(pin1, frames);
	strategy->push(pin2, frames);
	strategy->flush();

	BOOST_TEST(que1->size() == 1);
	BOOST_TEST(que2->size() == 1);


	strategy->push(pin1, frames);
	strategy->push(pin2, frames);
	strategy->flush();

	BOOST_TEST(que1->size() == 2);
	BOOST_TEST(que2->size() == 2);

	strategy->push(pin1, frames);
	strategy->push(pin2, frames);
	strategy->flush(); 

	// because que2 has max capacity of 2 .. so the push fails
	BOOST_TEST(que1->size() == 2);
	BOOST_TEST(que2->size() == 2);

	strategy->push(pin1, frames);	
	strategy->flush();

	// we pushed only for pin1 so it should be fine
	BOOST_TEST(que1->size() == 3);
	BOOST_TEST(que2->size() == 2);

	que2->pop();
	BOOST_TEST(que2->size() == 1);

	strategy->push(pin1, frames);
	strategy->push(pin2, frames);
	strategy->flush();

	BOOST_TEST(que1->size() == 4);
	BOOST_TEST(que2->size() == 2);
}


BOOST_AUTO_TEST_SUITE_END()