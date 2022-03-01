#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include "FileReaderModule.h"
#include "ExternalSinkModule.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "test_utils.h"
#include "PipeLine.h"
#include "AIPExceptions.h"
#include "FaceDetectorXform.h"
#include "ApraFaceInfo.h"
#include "FaceDetectsInfo.h"
#include "ImageDecoderCV.h"

BOOST_AUTO_TEST_SUITE(facedetector_tests)

BOOST_AUTO_TEST_CASE(basic)
{
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/faces.jpg")));
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	fileReader->addOutputPin(metadata);

	auto decoder = boost::shared_ptr<ImageDecoderCV>(new ImageDecoderCV(ImageDecoderCVProps()));
    auto metadata2 = framemetadata_sp(new RawImageMetadata());
    decoder->addOutputPin(metadata2);
	fileReader->setNext(decoder);

	FaceDetectorXformProps faceDetectorProps;
	auto faceDetector = boost::shared_ptr<FaceDetectorXform>(new FaceDetectorXform(faceDetectorProps));
	decoder->setNext(faceDetector);
	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	faceDetector->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(decoder->init());
	BOOST_TEST(faceDetector->init());
	BOOST_TEST(sink->init());

	fileReader->step();
	decoder->step();
	faceDetector->step();
	auto frames = sink->pop();
	BOOST_TEST(frames.size() == 1);
	FaceDetectsInfo result = FaceDetectsInfo::deSerialize(frames);
	BOOST_TEST(result.facesFound);
	cv::Mat frame = cv::imread("./data/faces.jpg");
	for (int i = 0; i < result.faces.size(); i++)
	{
		auto face = result.faces[i];
		cv::Point pt1(face.x1, face.y1);
		cv::Point pt2(face.x2, face.y2);
		cv::rectangle(frame, pt1, pt2, cv::Scalar(0, 255, 0), 2);
	}
	Test_Utils::saveOrCompare("./data/testOutput/facesResult.raw", const_cast<const uint8_t *>(static_cast<uint8_t *>(frame.data)), frame.step[0] * frame.rows, 0);
}

BOOST_AUTO_TEST_CASE(serialization_test)
{

	FaceDetectsInfo result;
	ApraFaceInfo face;
	face.x1 = 1.0;
	face.y1 = 1.0;
	face.x2 = 1.0;
	face.y2 = 1.0;
	face.score = 1.0;
	std::vector<ApraFaceInfo> apraFaces;
	apraFaces.emplace_back(face);
	result.faces = apraFaces;
	auto size = result.getSerializeSize();
	auto ptr = new uint8_t[size];
	memset(ptr, size, 0);
	result.serialize((void *)ptr, size);

	FaceDetectsInfo result2;
	Utils::deSerialize<FaceDetectsInfo>(result2, ptr, size);

	BOOST_TEST(result2.faces.size() == result.faces.size());
	BOOST_TEST(result2.faces[0].x1 == result.faces[0].x1);
	BOOST_TEST(result2.faces[0].y1 == result.faces[0].y1);
	BOOST_TEST(result2.faces[0].x2 == result.faces[0].x2);
	BOOST_TEST(result2.faces[0].y2 == result.faces[0].y2);
	BOOST_TEST(result2.facesFound);
	BOOST_TEST(result2.faces[0].score == result.faces[0].score);

	delete ptr;
}

BOOST_AUTO_TEST_CASE(aprafaceinfo_test)
{

	ApraFaceInfo face;
	face.x1 = 1.0;
	face.y1 = 1.0;
	face.x2 = 1.0;
	face.y2 = 1.0;
	face.score = 1.0;
	auto size = face.getSerializeSize();
	auto ptr = new uint8_t[size];
	memset(ptr, size, 0);
	Utils::serialize<ApraFaceInfo>(face, ptr, size);

	ApraFaceInfo face2;
	Utils::deSerialize<ApraFaceInfo>(face2, ptr, size);


	BOOST_TEST(face.x1 == face2.x1);
	BOOST_TEST(face.y1 == face2.y1);
	BOOST_TEST(face.x2 == face2.x2);
	BOOST_TEST(face.y2 == face2.y2);
	BOOST_TEST(face.score == face2.score);

	delete ptr;
}

BOOST_AUTO_TEST_SUITE_END()