#include <boost/test/unit_test.hpp>

#include "FileReaderModule.h"
#include "ExternalSinkModule.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "AIPExceptions.h"
#include "FacialLandmarkCV.h"
#include "test_utils.h"
#include <opencv2/core/types.hpp>
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

private:
	std::string m_log;

public:
	ExternalSink(ExternalSinkProps props) : Module(SINK, "ExternalSink", props)
	{
	}

	std::string getLog() const
	{
		return m_log;
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
		for (const auto& pair : frames)
		{
			LOG_INFO << pair.first << "," << pair.second;
		}

		auto frame = Module::getFrameByType(frames, FrameMetadata::FrameType::RAW_IMAGE);
		if (frame)
			LOG_INFO << "Timestamp <" << frame->timestamp << ">";

		vector<vector<ApraPoint2f>> facelandmarks;
		Utils::deSerialize<std::vector<std::vector<ApraPoint2f>>>(facelandmarks, frame->data(), frame->size());

		for (int i = 0; i < facelandmarks.size(); i++)
		{
			m_log += "Facial landmarks for face #" + std::to_string(i) + ":\n";
			for (int j = 0; j < facelandmarks[i].size(); j++)
			{
				ApraPoint2f p(facelandmarks[i][j]);
				m_log += "Point #" + std::to_string(j) + ": (" + std::to_string(p.x) + "," + std::to_string(p.y) + ")\n";
			}
		}

		cv::Mat iImg = cv::imread("./data/tilt.jpg");

		for (int i = 0; i < facelandmarks.size(); i++)
		{
			for (int j = 0; j < facelandmarks[i].size(); j++)
			{
				ApraPoint2f p(facelandmarks[i][j]);
				circle(iImg, p, 3, cv::Scalar(255, 200, 0), cv::LineTypes::FILLED);
			}
		}
		
		cv::imwrite("./data/landamrksplot.png", iImg);
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

	auto facemark = boost::shared_ptr<FacialLandmarkCV>(new FacialLandmarkCV(FacialLandmarkCVProps(FacialLandmarkCVProps::FaceDetectionModelType::HAAR_CASCADE)));
	fileReader->setNext(facemark);

	auto sink = boost::shared_ptr<ExternalSink>(new ExternalSink(ExternalSinkProps()));
	facemark->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(facemark->init());
	BOOST_TEST(sink->init());

	fileReader->step();
	facemark->step();
	sink->step();
	std::string allLogs = sink->getLog();
}

BOOST_AUTO_TEST_CASE(tilted_face)
{
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/tilt.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(1300,1151, ImageMetadata::ImageType::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));
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
	std::string allLogs = sink->getLog();
}

BOOST_AUTO_TEST_SUITE_END()
