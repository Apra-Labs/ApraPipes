#include <boost/test/unit_test.hpp>
#include <thread>
#include <boost/thread.hpp>
#include <boost/chrono/chrono.hpp>
#include "GPIODriver.h"

BOOST_AUTO_TEST_SUITE(gpiodriver_tests, * boost::unit_test::disabled())

int read()
{
    GPIODriver driver(388);
    BOOST_TEST(driver.Init(true));

    auto value = driver.Read();

    return value;
}

void write(bool high)
{
    GPIODriver driver(297);
    BOOST_TEST(driver.Init(false));

    BOOST_TEST(driver.Write(high));
}

BOOST_AUTO_TEST_CASE(read_write_input, *boost::unit_test::disabled())
{
    write(true);
    BOOST_TEST(read() == 1);
    write(false);
    BOOST_TEST(read() == 0);
    write(true);
    BOOST_TEST(read() == 1);
}

void writeforinterrupttest()
{
    GPIODriver driver(297);
    BOOST_TEST(driver.Init(false));
    BOOST_TEST(driver.Write(false));

    boost::this_thread::sleep_for(boost::chrono::seconds(1));

    BOOST_TEST(driver.Write(true));
    boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
    BOOST_TEST(driver.Write(false));
    boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
    BOOST_TEST(driver.Write(true));
    boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
    BOOST_TEST(driver.Write(false));
    boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
    BOOST_TEST(driver.Write(true));
    boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
    BOOST_TEST(driver.Write(false));
    boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
}

BOOST_AUTO_TEST_CASE(ReadWithInterrupt, *boost::unit_test::disabled())
{    
    std::thread myThread(writeforinterrupttest);	

    boost::this_thread::sleep_for(boost::chrono::milliseconds(200));

    GPIODriver driver(388);
    BOOST_TEST(driver.Init4EdgeInterrupt(2*1000));
    BOOST_TEST(driver.ReadWithInterrupt() == 0); // initially 0
    BOOST_TEST(driver.ReadWithInterrupt() == 1);
    BOOST_TEST(driver.ReadWithInterrupt() == 0);
    BOOST_TEST(driver.ReadWithInterrupt() == 1);
    BOOST_TEST(driver.ReadWithInterrupt() == 0);
    BOOST_TEST(driver.ReadWithInterrupt() == 1);
    BOOST_TEST(driver.ReadWithInterrupt() == 0);
    BOOST_TEST(driver.ReadWithInterrupt() == -1);

    myThread.join();
}

BOOST_AUTO_TEST_SUITE_END()
