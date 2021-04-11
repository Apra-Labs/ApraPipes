#include "ApraEGLDisplay.h"

#include <boost/test/unit_test.hpp>

// NOTE: TESTS WHICH REQUIRE ANY ENVIRONMENT TO BE PRESENT BEFORE RUNNING ARE NOT UNIT TESTS !!!

BOOST_AUTO_TEST_SUITE(apraegldisplay_tests)

BOOST_AUTO_TEST_CASE(basic)
{
	ApraEGLDisplay::getEGLDisplay();

}

BOOST_AUTO_TEST_SUITE_END()