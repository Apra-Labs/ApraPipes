/**
 * @file test_runner.cpp
 * @brief Main test runner for all ApraPipes samples unit tests
 *
 * This file serves as the entry point for running all sample tests.
 * It uses Boost.Test framework to discover and run all test suites.
 */

#define BOOST_TEST_MODULE ApraPipesSamplesTests
#include <boost/test/unit_test.hpp>

// The test runner will automatically discover all BOOST_AUTO_TEST_SUITE tests
// from the included test files

/**
 * Global test fixture for samples
 * This runs before all tests and after all tests
 */
struct GlobalTestFixture {
    GlobalTestFixture() {
        BOOST_TEST_MESSAGE("Starting ApraPipes Samples Unit Tests");
    }

    ~GlobalTestFixture() {
        BOOST_TEST_MESSAGE("Finished ApraPipes Samples Unit Tests");
    }
};

BOOST_GLOBAL_FIXTURE(GlobalTestFixture);
