#include <boost/test/unit_test.hpp>

#include "face_detection_cpu.h"

BOOST_AUTO_TEST_SUITE(test_face_detection_cpu)

BOOST_AUTO_TEST_CASE(face_detection_cpu)
{
    auto faceDetectionCPUPipeline = boost::shared_ptr<FaceDetectionCPU>(new FaceDetectionCPU());
    faceDetectionCPUPipeline->setupPipeline();
    faceDetectionCPUPipeline->startPipeline();

    boost::this_thread::sleep_for(boost::chrono::seconds(50));

    faceDetectionCPUPipeline->stopPipeline();
}

BOOST_AUTO_TEST_SUITE_END()