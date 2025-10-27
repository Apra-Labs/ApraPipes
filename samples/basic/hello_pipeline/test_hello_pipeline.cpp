/**
 * @file test_hello_pipeline.cpp
 * @brief Unit tests for hello_pipeline sample
 */

#include <boost/test/unit_test.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>

// ApraPipes modules
#include "PipeLine.h"
#include "TestSignalGeneratorSrc.h"
#include "StatSink.h"

BOOST_AUTO_TEST_SUITE(hello_pipeline_tests)

/**
 * @brief Test basic pipeline setup and initialization
 */
BOOST_AUTO_TEST_CASE(test_pipeline_setup)
{
    // Create a simple test pipeline
    auto source = boost::shared_ptr<TestSignalGeneratorSrc>(
        new TestSignalGeneratorSrc(TestSignalGeneratorSrcProps())
    );

    auto sink = boost::shared_ptr<StatSink>(
        new StatSink(StatSinkProps())
    );

    source->setNext(sink);

    PipeLine pipeline("test_hello_pipeline");
    pipeline.appendModule(source);

    // Test initialization
    BOOST_CHECK_NO_THROW(pipeline.init());

    // Test starting pipeline
    BOOST_CHECK_NO_THROW(pipeline.run_all_threaded());

    // Let it run briefly
    boost::this_thread::sleep_for(boost::chrono::milliseconds(100));

    // Test stopping pipeline
    BOOST_CHECK_NO_THROW(pipeline.stop());
    BOOST_CHECK_NO_THROW(pipeline.term());
    BOOST_CHECK_NO_THROW(pipeline.wait_for_all());
}

/**
 * @brief Test pipeline lifecycle (init -> run -> stop)
 */
BOOST_AUTO_TEST_CASE(test_pipeline_lifecycle)
{
    auto source = boost::shared_ptr<TestSignalGeneratorSrc>(
        new TestSignalGeneratorSrc(TestSignalGeneratorSrcProps())
    );

    auto sink = boost::shared_ptr<StatSink>(
        new StatSink(StatSinkProps())
    );

    source->setNext(sink);

    PipeLine pipeline("test_lifecycle");
    pipeline.appendModule(source);

    // Multiple start/stop cycles
    for (int i = 0; i < 3; i++) {
        BOOST_CHECK_NO_THROW(pipeline.init());
        BOOST_CHECK_NO_THROW(pipeline.run_all_threaded());
        boost::this_thread::sleep_for(boost::chrono::milliseconds(50));
        BOOST_CHECK_NO_THROW(pipeline.stop());
        BOOST_CHECK_NO_THROW(pipeline.term());
        BOOST_CHECK_NO_THROW(pipeline.wait_for_all());
    }
}

BOOST_AUTO_TEST_SUITE_END()
