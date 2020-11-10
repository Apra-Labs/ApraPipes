#include <boost/test/unit_test.hpp>

#include "ExternalSourceModule.h"
#include "GPIOSink.h"
#include "GPIODriver.h"
#include "GPIOMetadata.h"
#include "FrameMetadata.h"
#include "Frame.h"

#include <condition_variable>
#include <mutex>
#include <atomic>

BOOST_AUTO_TEST_SUITE(gpiosink_tests, * boost::unit_test::disabled())

mutex mutex_;
condition_variable con1, con2;
atomic_bool started{true};

void readhelper(GPIODriver &driver, int value, std::string message)
{
    unique_lock<mutex> lock(mutex_);
    con2.wait(lock, [] { return started.load(); });
    auto actualValue = driver.ReadWithInterrupt();
    std::cout << actualValue << "<>" << value << std::endl;
    BOOST_TEST(actualValue == value, message);
    started.store(false);
    lock.unlock();
    con1.notify_one();
}

void readvalues()
{
    GPIODriver driver(388);
    BOOST_TEST(driver.Init4EdgeInterrupt(2 * 1000));
    readhelper(driver, 0, "v1"); // initially 0

    readhelper(driver, 1, "v2");
    BOOST_TEST(driver.ReadWithInterrupt() == 0);
    readhelper(driver, 1, "v3");
    BOOST_TEST(driver.ReadWithInterrupt() == 0);
    readhelper(driver, 1, "v4");
    BOOST_TEST(driver.ReadWithInterrupt() == 0);
    BOOST_TEST(driver.ReadWithInterrupt() == -1);
}

BOOST_AUTO_TEST_CASE(basic)
{
    auto source = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
    auto metadata = framemetadata_sp(new GPIOMetadata());
    auto pinId = source->addOutputPin(metadata);

    auto sink = boost::shared_ptr<Module>(new GPIOSink(GPIOSinkProps(297, 1000)));
    source->setNext(sink);

    BOOST_TEST(source->init());
    BOOST_TEST(sink->init());

    auto frame = source->makeFrame(metadata->getDataSize(), metadata);
    frame_container frames;
    frames.insert(make_pair(pinId, frame));

    std::thread myThread(readvalues);
    boost::this_thread::sleep_for(boost::chrono::milliseconds(200));

    for (auto i = 0; i < 3; i++)
    {
        unique_lock<mutex> lock(mutex_);
        con1.wait(lock, [] { return !started.load(); });
        started.store(true);
        lock.unlock();
        con2.notify_one();

        source->send(frames);
        sink->step();
        boost::this_thread::sleep_for(boost::chrono::milliseconds(500));
    }

    myThread.join();
}

void readvaluessetpropsnopin()
{
    GPIODriver driver(388);
    BOOST_TEST(driver.Init4EdgeInterrupt(2 * 100));
    driver.ReadWithInterrupt(); // don't care what the initial value is

    readhelper(driver, -1, "v2");
    BOOST_TEST(driver.ReadWithInterrupt() == -1);
    readhelper(driver, -1, "v3");
    BOOST_TEST(driver.ReadWithInterrupt() == -1);
    readhelper(driver, -1, "v4");
    BOOST_TEST(driver.ReadWithInterrupt() == -1);
    BOOST_TEST(driver.ReadWithInterrupt() == -1);
}

void readvaluessetprops()
{
    GPIODriver driver(388);
    BOOST_TEST(driver.Init4EdgeInterrupt(2 * 1000));
    readhelper(driver, 0, "v1"); // initially 0
   
    readhelper(driver, 1, "v2");
    BOOST_TEST(driver.ReadWithInterrupt() == 0);
    readhelper(driver, 1, "v3");
    BOOST_TEST(driver.ReadWithInterrupt() == 0);
    readhelper(driver, 1, "v4");
    BOOST_TEST(driver.ReadWithInterrupt() == 0);
    BOOST_TEST(driver.ReadWithInterrupt() == -1);
}

BOOST_AUTO_TEST_CASE(setprops)
{
    auto source = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
    auto metadata = framemetadata_sp(new GPIOMetadata());
    auto pinId = source->addOutputPin(metadata);

    auto sink = boost::shared_ptr<GPIOSink>(new GPIOSink(GPIOSinkProps(0, 1000)));
    source->setNext(sink);

    BOOST_TEST(source->init());
    BOOST_TEST(sink->init());

    auto frame = source->makeFrame(metadata->getDataSize(), metadata);
    frame_container frames;
    frames.insert(make_pair(pinId, frame));

    {
        std::thread myThread(readvaluessetpropsnopin);
        boost::this_thread::sleep_for(boost::chrono::milliseconds(200));

        for (auto i = 0; i < 3; i++)
        {
            unique_lock<mutex> lock(mutex_);
            con1.wait(lock, [] { return !started.load(); });
            started.store(true);
            lock.unlock();
            con2.notify_one();

            source->send(frames);
            sink->step();
            boost::this_thread::sleep_for(boost::chrono::milliseconds(500));
        }

        myThread.join();
    }

    GPIOSinkProps props(297, 1000);
    sink->setProps(props);
    sink->step();

    started.store(false);
    std::thread myThread(readvaluessetprops);
    boost::this_thread::sleep_for(boost::chrono::milliseconds(200));

    for (auto i = 0; i < 3; i++)
    {
        unique_lock<mutex> lock(mutex_);
        con1.wait(lock, [] { return !started.load(); });
        started.store(true);
        lock.unlock();
        con2.notify_one();

        source->send(frames);
        sink->step();
        boost::this_thread::sleep_for(boost::chrono::milliseconds(500));
    }

    myThread.join();
}

BOOST_AUTO_TEST_SUITE_END()