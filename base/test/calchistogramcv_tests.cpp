#include "stdafx.h"
#include <boost/test/unit_test.hpp>

#include "Module.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "CalcHistogramCV.h"
#include "AIPExceptions.h"
#include "ExternalSinkModule.h"
#include "ExternalSourceModule.h"
#include "FileReaderModule.h"
#include "Logger.h"

BOOST_AUTO_TEST_SUITE(calchistogramcv_tests, * boost::unit_test::disabled())

BOOST_AUTO_TEST_CASE(calchistogramcv_basic)
{
	{			
		auto m1 = boost::shared_ptr<Module>(new ExternalSourceModule());
		auto metadata = framemetadata_sp(new RawImageMetadata(100, 100, 1, CV_8UC1, 100, CV_8U));		
		m1->addOutputPin(metadata);

		CalcHistogramCVProps histProps(8);
		auto m2 = boost::shared_ptr<Module>(new CalcHistogramCV(histProps));
		BOOST_TEST(m2->init() == false);
		m1->setNext(m2);
		BOOST_TEST(m2->init() == false);
		auto histMetadata = framemetadata_sp(new ArrayMetadata());
		m2->addOutputPin(histMetadata);
		
		BOOST_TEST(m2->init());
	} 

	{			
		auto m1 = boost::shared_ptr<Module>(new ExternalSourceModule());
		auto metadata = framemetadata_sp(new RawImageMetadata());		
		m1->addOutputPin(metadata);

		CalcHistogramCVProps histProps(8);
		histProps.roi = { 491,6,1429, 338 };		
		auto m2 = boost::shared_ptr<Module>(new CalcHistogramCV(histProps));
		m1->setNext(m2);
		auto histMetadata = framemetadata_sp(new ArrayMetadata());
		m2->addOutputPin(histMetadata);

		BOOST_TEST(m2->init());
	}

	{		
		auto m1 = boost::shared_ptr<Module>(new ExternalSourceModule());
		auto metadata = framemetadata_sp(new RawImageMetadata());		
		m1->addOutputPin(metadata);
				
		CalcHistogramCVProps histProps(8);
		histProps.maskImgPath = "./data/maskImg.jpg";
		auto m2 = boost::shared_ptr<Module>(new CalcHistogramCV(histProps));
		m1->setNext(m2);
		auto histMetadata = framemetadata_sp(new ArrayMetadata());
		m2->addOutputPin(histMetadata);

		BOOST_TEST(m2->init());
	}

	{		
		auto m1 = boost::shared_ptr<Module>(new ExternalSourceModule());
		auto metadata = framemetadata_sp(new RawImageMetadata(100, 100, 1, CV_8UC1, 100, CV_8U));		
		m1->addOutputPin(metadata);

		CalcHistogramCVProps histProps(8);
		histProps.roi = { 491,6,1429, 3338 };
		auto m2 = boost::shared_ptr<Module>(new CalcHistogramCV(histProps));
		m1->setNext(m2);
		auto histMetadata = framemetadata_sp(new ArrayMetadata());
		m2->addOutputPin(histMetadata);

		try
		{
			m2->init();
			BOOST_TEST(false);
		}
		catch (AIP_Exception& exception)
		{
			BOOST_TEST(exception.getCode() == AIP_ROI_OUTOFRANGE);
		}
		catch (...)
		{
			BOOST_TEST(false);
		}
	}

	{
		auto m1 = boost::shared_ptr<Module>(new ExternalSourceModule());
		auto metadata = framemetadata_sp(new RawImageMetadata(100, 100, 1, CV_8UC1, 100, CV_8U));
		m1->addOutputPin(metadata);
				
		try
		{
			CalcHistogramCVProps histProps(8);
			histProps.maskImgPath = "maskImg.jpg";
			auto m2 = boost::shared_ptr<Module>(new CalcHistogramCV(histProps));
			BOOST_TEST(false);
		}
		catch (AIP_Exception& exception)
		{
			BOOST_TEST(exception.getCode() == AIP_IMAGE_LOAD_FAILED);
		}
		catch (...)
		{
			BOOST_TEST(false);
		}
	}

	{
		auto m1 = boost::shared_ptr<Module>(new ExternalSourceModule());
		auto metadata = framemetadata_sp(new RawImageMetadata(100, 100, 1, CV_8UC1, 100, CV_8U));
		m1->addOutputPin(metadata);

		CalcHistogramCVProps histProps(8);
		histProps.maskImgPath = "./data/maskImg.jpg";
		auto m2 = boost::shared_ptr<Module>(new CalcHistogramCV(histProps));
		m1->setNext(m2);
		auto histMetadata = framemetadata_sp(new ArrayMetadata());
		m2->addOutputPin(histMetadata);

		try
		{
			m2->init();
			BOOST_TEST(false);
		}
		catch (AIP_Exception& exception)
		{
			BOOST_TEST(exception.getCode() == AIP_ROI_OUTOFRANGE);
		}
		catch (...)
		{
			BOOST_TEST(false);
		}
	}

	{
		// process SOS will be called and it checks for roi range. Exception has to be thrown
		cv::Mat img = cv::imread("./data/frame.jpg");
		if (!img.data)
		{
			BOOST_TEST(false);
		}

		auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
		auto metadata = framemetadata_sp(new RawImageMetadata());		
		auto rawImagePinId = m1->addOutputPin(metadata);

		CalcHistogramCVProps props;
		props.roi = { 491,6,1429, 3338 };
		auto m2 = boost::shared_ptr<Module>();
		m2.reset(new CalcHistogramCV(props));

		m1->setNext(m2);
		auto histMetadata = framemetadata_sp(new ArrayMetadata());
		auto histPinId = m2->addOutputPin(histMetadata);


		auto m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
		m2->setNext(m3);

		BOOST_TEST(m1->init());
		BOOST_TEST(m2->init());
		BOOST_TEST(m3->init());

		FrameMetadataFactory::downcast<RawImageMetadata>(metadata)->setData(img);
		auto rawImageFrame = m1->makeFrame(metadata->getDataSize(), metadata);
		memcpy(rawImageFrame->data(), img.data, metadata->getDataSize());

		frame_container frames;
		frames.insert(make_pair(rawImagePinId, rawImageFrame));
		m1->send(frames);

		m1->send(frames);
		try
		{
			m2->step();
			BOOST_TEST(false);
		}
		catch (AIP_Exception& exception)
		{
			BOOST_TEST(exception.getCode() == AIP_ROI_OUTOFRANGE);
		}
		catch (...)
		{
			BOOST_TEST(false);
		}

	}

}

vector<float> histValuesRoi = {
  0.0025631363845284283f,
  0.28254955466022913f,
  0.02093159034538159f,
  0.2167030364263502f,
  0.4771056848625886f,
  0.00014699732092206657f,
  0.0f,
  0.0f
};

vector<float> histValuesRoi_16 = { 0.0f,0.0025631363845284283f,0.16177779802154027f,0.12077175663868887f,0.010229771305294802f,0.010701819040086791f,0.06521505086935458f,0.15148798555699564f,0.34146442457795206f,0.1356412602846365f,0.00014699732092206657f,0.0f,0.0f,0.0f,0.0f,0.0f };

vector<float> histValuesNone = {
  0.04771819933920705f,
  0.3370170245961821f,
  0.06624334618208517f,
  0.17092855176211455f,
  0.32647072320117476f,
  0.048526982378854625f,
  0.003005690161527166f,
  8.948237885462555e-05f
};

vector<float> histValuesNone_16 = { 0.0f,0.04771819933920705f,0.20659760462555066f,0.13041941997063142f,0.03153221365638766f,0.034711132525697505f,0.06225334985315712f,0.10867520190895742f,0.21675270741556535f,0.1097180157856094f,0.03671530837004405f,0.011811674008810573f,0.0024814151982378856f,0.0005242749632892805f,8.948237885462555e-05f,0.0f };

vector<float> histValuesMaskImg = {
	0.002813546306885035f, 0.28233101176198594f, 0.022886124654757433f, 0.21613526490271281f, 0.47568749200110644f, 0.00014656037255233857f, 0.0f, 0.0f
};

vector<float> histValuesMaskImg_16 = { 0.0f,0.002813546306885035f,0.16183567898737103f,0.12049533277461492f,0.010622530664145553f,0.012263593990611879f,0.06509757618043027f,0.15103768872228254f,0.3404494242860858f,0.1352380677150206f,0.00014656037255233857f,0.0f,0.0f,0.0f,0.0f,0.0f };

void testValues(int bins, int type, vector<float>& histValues)
{
	cv::Mat img = cv::imread("./data/frame.jpg");
	if (!img.data)
	{
		BOOST_TEST(false);
	}

	auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new RawImageMetadata());
	FrameMetadataFactory::downcast<RawImageMetadata>(metadata)->setData(img);
	auto rawImagePinId = m1->addOutputPin(metadata);
		
	CalcHistogramCVProps props(bins);
	if (type == 2)
	{		
		props.roi = { 491,6,1429, 338 };		
	}
	else if (type == 3)
	{
		props.maskImgPath = "./data/maskImg.jpg";
	}
	auto m2 = boost::shared_ptr<Module>(new CalcHistogramCV(props));
	m1->setNext(m2);
	auto histMetadata = framemetadata_sp(new ArrayMetadata());
	auto histPinId = m2->addOutputPin(histMetadata);
	
	auto m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	m2->setNext(m3);

	BOOST_TEST(m1->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(m3->init());

	auto rawImageFrame = m1->makeFrame(metadata->getDataSize(), metadata);
	memcpy(rawImageFrame->data(), img.data, metadata->getDataSize());

	frame_container frames;
	frames.insert(make_pair(rawImagePinId, rawImageFrame));
	m1->send(frames);


	m2->step();
	auto outputFrames = m3->pop();
	BOOST_TEST((outputFrames.find(histPinId) != outputFrames.end()));
	auto histFrame = outputFrames[histPinId];	

	cv::Mat outImg(bins, 1, CV_32FC1, histFrame->data());
	for (auto i = 0; i < bins; i++)
	{					
		auto actualVal = outImg.at<float>(i, 0);
		auto expectedVal = histValues[i];		
		BOOST_TEST(expectedVal == actualVal, boost::test_tools::tolerance(0.00001f));
	}
}

BOOST_AUTO_TEST_CASE(calchistogramcv_values_basic)
{
	testValues(8, 2, histValuesRoi);
	testValues(8, 1, histValuesNone);
	testValues(8, 3, histValuesMaskImg);
}

void testValues2(frame_container& frames, boost::shared_ptr<ExternalSourceModule>& m1, boost::shared_ptr<CalcHistogramCV>& m2, boost::shared_ptr<ExternalSinkModule>& m3, string& histPinId, vector<float>& histValues)
{
	m1->send(frames);


	m2->step();
	auto outputFrames = m3->pop();
	BOOST_TEST((outputFrames.find(histPinId) != outputFrames.end()));
	auto histFrame = outputFrames[histPinId];
	auto histMetadata = histFrame->getMetadata();

	auto props = m2->getProps();
	auto bins = props.bins;	

	cv::Mat outImg(bins, 1, CV_32FC1, histFrame->data());
	for (auto i = 0; i < bins; i++)
	{
		auto actualVal = outImg.at<float>(i, 0);
		auto expectedVal = histValues[i];
		BOOST_TEST(expectedVal == actualVal, boost::test_tools::tolerance(0.00001f));
	}
	
}

void testSetProps(int bins, int type, boost::shared_ptr<CalcHistogramCV>& m2, boost::shared_ptr<ExternalSinkModule>& m3, string& histPinId )
{
	auto props = m2->getProps();
	props.bins = bins;
	props.roi = {};
	props.maskImgPath = "";
	if (type == 2)
	{
		props.roi = { 491,6,1429, 338 };
	} 
	else if (type == 3)
	{
		props.maskImgPath = "./data/maskImg.jpg";
	}

	m2->setProps(props);
	m2->step();

	props = m2->getProps();
	BOOST_TEST(props.bins == bins);
	if (type == 2)
	{
		BOOST_TEST(props.roi.size() == 4);
		BOOST_TEST(props.roi[0] == 491);
		BOOST_TEST(props.roi[1] == 6);
		BOOST_TEST(props.roi[2] == 1429);
		BOOST_TEST(props.roi[3] == 338);
	}
	else
	{
		BOOST_TEST(props.roi.size() == 0);
	}

	if (type == 3)
	{
		BOOST_TEST(props.maskImgPath == "./data/maskImg.jpg");
	}
	else
	{
		BOOST_TEST(props.maskImgPath == "");
	}

	auto outputFrames = m3->pop();
	BOOST_TEST((outputFrames.find(histPinId) != outputFrames.end()));
	auto eosFrame = outputFrames[histPinId];
	BOOST_TEST(eosFrame->isEOS());
}

BOOST_AUTO_TEST_CASE(calchistogramcv_values_withchangeprops)
{
	cv::Mat img = cv::imread("./data/frame.jpg");
	if (!img.data)
	{
		BOOST_TEST(false);
	}

	auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new RawImageMetadata());
	FrameMetadataFactory::downcast<RawImageMetadata>(metadata)->setData(img);
	auto rawImagePinId = m1->addOutputPin(metadata);

	int bins = 8;
	CalcHistogramCVProps props(bins);	
	auto m2 = boost::shared_ptr<CalcHistogramCV>(new CalcHistogramCV(props));
	m1->setNext(m2);
	auto histMetadata = framemetadata_sp(new ArrayMetadata());
	auto histPinId = m2->addOutputPin(histMetadata);
		
	auto m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	m2->setNext(m3);

	BOOST_TEST(m1->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(m3->init());

	auto rawImageFrame = m1->makeFrame(metadata->getDataSize(), metadata);
	memcpy(rawImageFrame->data(), img.data, metadata->getDataSize());

	frame_container frames;
	frames.insert(make_pair(rawImagePinId, rawImageFrame));

	testValues2(frames, m1, m2, m3, histPinId, histValuesNone);
	testSetProps(16, 1, m2, m3, histPinId);
	testValues2(frames, m1, m2, m3, histPinId, histValuesNone_16);
	testSetProps(8, 2, m2, m3, histPinId);
	testValues2(frames, m1, m2, m3, histPinId, histValuesRoi);
	testSetProps(16, 2, m2, m3, histPinId);
	testValues2(frames, m1, m2, m3, histPinId, histValuesRoi_16);
	testSetProps(16, 3, m2, m3, histPinId);
	testValues2(frames, m1, m2, m3, histPinId, histValuesMaskImg_16);
	testSetProps(8, 3, m2, m3, histPinId);
	testValues2(frames, m1, m2, m3, histPinId, histValuesMaskImg);	
	testSetProps(16, 1, m2, m3, histPinId);
	testValues2(frames, m1, m2, m3, histPinId, histValuesNone_16);
	testSetProps(8, 2, m2, m3, histPinId);
	testValues2(frames, m1, m2, m3, histPinId, histValuesRoi);
	testSetProps(16, 1, m2, m3, histPinId);
	testValues2(frames, m1, m2, m3, histPinId, histValuesNone_16);
}

BOOST_AUTO_TEST_CASE(calchistogramcv_perf)
{
	return;

	cv::Mat img = cv::imread("./data/frame.jpg");
	if (!img.data)
	{
		BOOST_TEST(false);
	}

	auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new RawImageMetadata());
	FrameMetadataFactory::downcast<RawImageMetadata>(metadata)->setData(img);
	auto rawImagePinId = m1->addOutputPin(metadata);

	auto m2 = boost::shared_ptr<Module>();
	m2.reset(new CalcHistogramCV(CalcHistogramCVProps()));
	
	m1->setNext(m2);
	auto histMetadata = framemetadata_sp(new ArrayMetadata());
	auto histPinId = m2->addOutputPin(histMetadata);
	
	auto m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	m2->setNext(m3);

	BOOST_TEST(m1->init());
	BOOST_TEST(m2->init());
	BOOST_TEST(m3->init());

	auto rawImageFrame = m1->makeFrame(metadata->getDataSize(), metadata);
	memcpy(rawImageFrame->data(), img.data, metadata->getDataSize());

	frame_container frames;
	frames.insert(make_pair(rawImagePinId, rawImageFrame));
	m1->send(frames);

	auto j = 0;
	while (true)
	{
		using sys_clock = std::chrono::system_clock;
		sys_clock::time_point frame_begin = sys_clock::now();
		for (int i = 0; i < 1000; i++)
		{
			m1->send(frames);
			m2->step();
			m3->pop();			
		}
		sys_clock::time_point frame_end = sys_clock::now();
		std::chrono::nanoseconds frame_len = frame_end - frame_begin;
		LOG_ERROR << "loopindex<" << j << "> timeelapsed<" << 1.0*frame_len.count() / (1000000000.0) << "> fps<" << 1000 / (1.0*frame_len.count() / (1000000000.0)) << ">";
		j++;

		if (j == 10)
		{
			break;
		}
	}
}

BOOST_AUTO_TEST_SUITE_END()