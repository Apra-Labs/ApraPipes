#include <boost/test/unit_test.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/face.hpp>
#include "FileReaderModule.h"
#include "ExternalSinkModule.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "AIPExceptions.h"
#include "FacialLandmarksCV.h"
#include "test_utils.h"
#include "Frame.h"
#include "ApraPoint2f.h"

BOOST_AUTO_TEST_SUITE(faciallandmarkcv_tests)

class ExternalSinkProps : public ModuleProps
{
public:
	ExternalSinkProps() : ModuleProps()
	{
	}

};

class ExternalSink : public Module
{
public:
	ExternalSink(ExternalSinkProps props) : Module(SINK, "ExternalSink", props)
	{
	}

	frame_container pop()
	{
		return Module::pop();
	}

	boost::shared_ptr<FrameContainerQueue> getQue()
	{
		return Module::getQue();
	}

	~ExternalSink()
	{
	}

protected:
	bool process(frame_container& frames)
	{
		auto frame = Module::getFrameByType(frames, FrameMetadata::FrameType::FACE_LANDMARKS_INFO);

		vector<vector<ApraPoint2f>> facelandmarks;
		Utils::deSerialize<std::vector<std::vector<ApraPoint2f>>>(facelandmarks, frame->data(), frame->size());

		for (int i = 0; i < facelandmarks.size(); i++)
		{
		  assert(facelandmarks[i].size() == 68); // check if there are 68 facial landmarks
		}

		cv::Mat iImg = cv::imread("./data/faces.jpg");

		for (int i = 0; i < facelandmarks.size(); i++)
		{
			for (int j = 0; j < facelandmarks[i].size(); j++)
			{
				ApraPoint2f p(facelandmarks[i][j]);
				circle(iImg, p, 3, cv::Scalar(255, 200, 0), cv::LineTypes::FILLED);
			}
		}
		std::vector<uchar> buf;
		cv::imencode(".jpg", iImg, buf);

	    Test_Utils::saveOrCompare("./data/testOutput/facesResult.jpg", buf.data(), buf.size(), 0);
	    return true;
	}

	bool validateInputPins()
	{
		return true;
	}

	bool validateInputOutputPins()
	{
		return true;
	}

};

boost::shared_ptr<ExternalSink> sink;

BOOST_AUTO_TEST_CASE(multiple_faces)
{
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/faces.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(1024,768, ImageMetadata::ImageType::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));
	fileReader->addOutputPin(metadata);

	auto facemark = boost::shared_ptr<FacialLandmarkCV>(new FacialLandmarkCV(FacialLandmarkCVProps(FacialLandmarkCVProps::FaceDetectionModelType::SSD)));
	fileReader->setNext(facemark);

	auto sink = boost::shared_ptr<ExternalSink>(new ExternalSink(ExternalSinkProps()));
	facemark->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(facemark->init());
	BOOST_TEST(sink->init());

	fileReader->step();
	facemark->step();
	sink->step();
}

BOOST_AUTO_TEST_CASE(getSetProps)
{
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/faces.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(1024, 768, ImageMetadata::ImageType::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));
	fileReader->addOutputPin(metadata);

	auto facemark = boost::shared_ptr<FacialLandmarkCV>(new FacialLandmarkCV(FacialLandmarkCVProps(FacialLandmarkCVProps::FaceDetectionModelType::SSD)));
	fileReader->setNext(facemark);

	auto sink = boost::shared_ptr<ExternalSink>(new ExternalSink(ExternalSinkProps()));
	facemark->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(facemark->init());
	BOOST_TEST(sink->init());

	fileReader->step();
	facemark->step();

	auto type = FacialLandmarkCVProps::FaceDetectionModelType::SSD;
	auto currentProps = facemark->getProps();
	BOOST_ASSERT(type == currentProps.type);

	auto propsChange = FacialLandmarkCVProps(FacialLandmarkCVProps::FaceDetectionModelType::HAAR_CASCADE);
	facemark->setProps(propsChange);
	facemark->step();

	fileReader->step();
	facemark->step();
	sink->step();

}

BOOST_AUTO_TEST_CASE(SSD)
{
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/faces.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(1024, 768, ImageMetadata::ImageType::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));
	fileReader->addOutputPin(metadata);

	auto facemark = boost::shared_ptr<FacialLandmarkCV>(new FacialLandmarkCV(FacialLandmarkCVProps(FacialLandmarkCVProps::FaceDetectionModelType::SSD, "./data/assets/deploy.prototxt.txt", "./data/assets/res10_300x300_ssd_iter_140000_fp16.caffemodel", "./data/assets/face_landmark_model.dat", cv::face::FacemarkKazemi::create())));
	fileReader->setNext(facemark);

	auto sink = boost::shared_ptr<ExternalSink>(new ExternalSink(ExternalSinkProps()));
	facemark->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(facemark->init());
	BOOST_TEST(sink->init());

	fileReader->step();
	facemark->step();
	sink->step();
}

BOOST_AUTO_TEST_CASE(HAAR_CASCADE)
{
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/faces.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(1024, 768, ImageMetadata::ImageType::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));
	fileReader->addOutputPin(metadata);

	auto facemark = boost::shared_ptr<FacialLandmarkCV>(new FacialLandmarkCV(FacialLandmarkCVProps(FacialLandmarkCVProps::FaceDetectionModelType::HAAR_CASCADE,"./data/assets/haarcascade.xml", "./data/assets/face_landmark_model.dat", cv::face::FacemarkKazemi::create())));
	fileReader->setNext(facemark);

	auto sink = boost::shared_ptr<ExternalSink>(new ExternalSink(ExternalSinkProps()));
	facemark->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(facemark->init());
	BOOST_TEST(sink->init());

	fileReader->step();
	facemark->step();
	sink->step();
}
BOOST_AUTO_TEST_SUITE_END()
