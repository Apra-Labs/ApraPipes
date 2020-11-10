#include <boost/test/unit_test.hpp>

#include "ExternalSinkModule.h"
#include "GPIOSource.h"
#include "GPIODriver.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include <condition_variable>
#include <mutex>
#include <atomic>
#include <chrono>

using sys_clock = std::chrono::system_clock;


BOOST_AUTO_TEST_SUITE(gpiosource_tests)

mutex mutex_;
condition_variable con1, con2;
atomic_bool started{false}; 

void writehelper(GPIODriver& driver, bool high)
{
    unique_lock<mutex> lock(mutex_);
    con2.wait(lock, []{ return !started.load(); });    
    BOOST_TEST(driver.Write(high));
    started.store(true);
    lock.unlock();
    con1.notify_one();
}

void writeforinterrupttest()
{
    GPIODriver driver(297);
    BOOST_TEST(driver.Init(false));
    BOOST_TEST(driver.Write(false));
    
    writehelper(driver, true);
    writehelper(driver, false);
    writehelper(driver, true);
    writehelper(driver, false);
    writehelper(driver, true);
    writehelper(driver, false);
}

void testvalue(boost::shared_ptr<Module> source, boost::shared_ptr<ExternalSinkModule> sink, unsigned char expectedValue, std::string message)
{    
    unique_lock<mutex> lock(mutex_);
    con1.wait(lock, []{ return started.load(); });
    
    source->step();
	auto frames = sink->pop();
    BOOST_TEST(frames.size() == 1);
    auto frame = frames.begin()->second;
    auto buffer = (unsigned char *)frame->data();
    
    BOOST_TEST(buffer[0] == expectedValue, message);

    started.store(false);
    lock.unlock();
    con2.notify_one();
}

BOOST_AUTO_TEST_CASE(high_low)
{
    auto source = boost::shared_ptr<Module>(new GPIOSource(GPIOSourceProps(388)));
	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	source->setNext(sink);
		
	BOOST_TEST(source->init());
	BOOST_TEST(sink->init());

    auto prevValueTS = sys_clock::now();
    // 1 or 0 will come always first time
    source->step();
    auto frames = sink->pop();
    BOOST_TEST(frames.size() == 1);

    for (auto i = 0; i < 5; i++)
    {
        // timeout
        source->step();
        auto frames = sink->try_pop();
        BOOST_TEST(frames.size() == 0);
    }
    std::chrono::nanoseconds timeElapsed = sys_clock::now() - prevValueTS;
    bool valid = timeElapsed < std::chrono::nanoseconds(6*100*1000*1000); // 6 times poll - each timeout is 100 ms - converting to nanosecond
    BOOST_TEST(valid); 

    std::thread myThread(writeforinterrupttest);
    
    testvalue(source, sink, 1, "2");
    testvalue(source, sink, 0, "3");
    testvalue(source, sink, 1, "4");
    testvalue(source, sink, 0, "5");
    testvalue(source, sink, 1, "6");
    testvalue(source, sink, 0, "7");

    myThread.join();
    
}

void writehelperpulse(GPIODriver& driver, bool sleepTimeinMs)
{
    unique_lock<mutex> lock(mutex_);
    con2.wait(lock, []{ return !started.load(); });    
    BOOST_TEST(driver.Write(true));
    started.store(true);
    lock.unlock();
    con1.notify_one();

    if(sleepTimeinMs == 0)
    {
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    else
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(sleepTimeinMs));
    }    
    
    BOOST_TEST(driver.Write(false));
    boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
}


void writeforinterrupttestpulse()
{
    GPIODriver driver(297);
    BOOST_TEST(driver.Init(false));
    BOOST_TEST(driver.Write(false));
        
    writehelperpulse(driver, 1);
    writehelperpulse(driver, 1);
    writehelperpulse(driver, 1);
    
    writehelperpulse(driver, 10);

    writehelperpulse(driver, 0);
    writehelperpulse(driver, 0);

    writehelperpulse(driver, 1);
    writehelperpulse(driver, 1);

    writehelperpulse(driver, 0);
}

void testvaluepulse(boost::shared_ptr<Module> source, boost::shared_ptr<ExternalSinkModule> sink, std::string message)
{    
    unique_lock<mutex> lock(mutex_);
    con1.wait(lock, []{ return started.load(); });
    
    source->step();
	auto frames = sink->pop();
    BOOST_TEST(frames.size() == 1);
    auto frame = frames.begin()->second;
    auto buffer = (unsigned char *)frame->data();

    BOOST_TEST(buffer[0] == 1, message);
    
    source->step();
	frames = sink->try_pop();
    BOOST_TEST(frames.size() == 0, message);

    started.store(false);
    lock.unlock();
    con2.notify_one();
}

BOOST_AUTO_TEST_CASE(pulse)
{
    {
        GPIODriver driver(297);
        BOOST_TEST(driver.Init(false));
        BOOST_TEST(driver.Write(false));
    }

    auto source = boost::shared_ptr<Module>(new GPIOSource(GPIOSourceProps(388, 1)));
	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	source->setNext(sink);
		
	BOOST_TEST(source->init());
	BOOST_TEST(sink->init());    
    
    // read first time
    source->step();

    std::thread myThread(writeforinterrupttestpulse);
    
    testvaluepulse(source, sink, "pulse1");
    testvaluepulse(source, sink, "pulse2");
    testvaluepulse(source, sink, "pulse3");
    testvaluepulse(source, sink, "pulse4");
    testvaluepulse(source, sink, "pulse5");
    testvaluepulse(source, sink, "pulse6");
    testvaluepulse(source, sink, "pulse7");
    testvaluepulse(source, sink, "pulse8");
    testvaluepulse(source, sink, "pulse9");

    myThread.join();
}

BOOST_AUTO_TEST_SUITE_END()