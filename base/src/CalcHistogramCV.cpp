#include <opencv2/highgui.hpp>
#include <boost/foreach.hpp>

#include "CalcHistogramCV.h"
#include "Frame.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Utils.h"
#include "Logger.h"
#include "AIPExceptions.h"

class CalcHistogramCV::Detail
{

public:
	Detail(CalcHistogramCVProps& _props): mXOffset(0)
	{			
		range = new float[2];
		range[0] = 0;
		range[1] = 256;

		ranges = new const float*[1];
		ranges[0] = range;

		setProps(_props);
	};

	void setProps(CalcHistogramCVProps& _props)
	{
		bins = _props.bins;
		mProps.reset(new CalcHistogramCVProps(_props));
		numberOfPixels = 0;
		mRoi = cv::Rect(0, 0, 0, 0);
		maskImg = cv::Mat();

		// load roi // load mask Image
		if (_props.roi.size())
		{
			setRoi(_props.roi);		
		}	
		else if (!_props.maskImgPath.empty())
		{
			loadMaskImg(_props.maskImgPath);
		}
		
		// setInput histogram and inits 
		if (mOutputMetadata.get())
		{
			mOutputImg = Utils::getMatHeader(1, bins, FrameMetadataFactory::downcast<ArrayMetadata>(mOutputMetadata)->getType());
		}

		if (mInputMetadata.get())
		{
			setInputMetadata(mInputMetadata);
		}
	}

	~Detail() 
	{
		delete[] ranges;
		delete[] range;
	};
	
	void setRoi(vector<int>& roi)
	{
		if (roi.size() != 4)
		{			
			auto msg = "Specify roi with x, y, w, h. Received only<" + to_string(roi.size()) + ">";
			throw AIPException(AIP_WRONG_STRUCTURE, msg);
		}

		if (roi[0] < 0 || roi[1] < 0)
		{
			auto msg = "x and y has to be greater than zero.";
			throw AIPException(AIP_ROI_OUTOFRANGE,string(msg));
		}

		mRoi = cv::Rect(roi[0], roi[1], roi[2], roi[3]);
		numberOfPixels = roi[2] * roi[3];
	}	

	void loadMaskImg(string& maskImgPath)
	{
		maskImg = cv::imread(maskImgPath, cv::IMREAD_GRAYSCALE);
		if (!maskImg.data)
		{
			auto msg = "Failed to load mask image from <" + maskImgPath + ">";
			throw AIPException(AIP_IMAGE_LOAD_FAILED, msg);
		}		

		numberOfPixels = cv::countNonZero(maskImg);
	}

	void setInputMetadata(framemetadata_sp& metadata)
	{
		validateParams(metadata);
		mInputMetadata = metadata;

		auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
		mXOffset = 0;
		if (mRoi.width)
		{
			mXOffset = rawMetadata->getOffset(mRoi.x, mRoi.y);
			mInputImg = Utils::getMatHeader(mRoi, rawMetadata);
		}
		else
		{
			mInputImg = Utils::getMatHeader(rawMetadata);
		}		
	}
	
	void setOutputMetadata(framemetadata_sp& metadata)
	{
		mOutputMetadata = metadata;
		mOutputImg = Utils::getMatHeader(1, bins, FrameMetadataFactory::downcast<ArrayMetadata>(metadata)->getType());
	}

	framemetadata_sp getOutputMetadata()
	{
		return mOutputMetadata;
	}

	void compute(frame_sp& inFrame, frame_sp& outFrame)
	{
		setOutputImgData(outFrame);
		setInputImg(inFrame);
		calcHist(&mInputImg, 1, &channelNumber, maskImg, mOutputImg, 1, &bins, ranges, true, false);		
		mOutputImg = mOutputImg / numberOfPixels;		

		return;
	}

	bool shouldTriggerSOS()
	{		
		return mInputImg.rows == 0;
	}

	CalcHistogramCVProps getProps()
	{
		return *mProps.get();
	}
		
	int bins;

private:

	void validateParams(framemetadata_sp metadata)
	{
		auto rawImageMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
		if (numberOfPixels == 0)
		{
			numberOfPixels = rawImageMetadata->getWidth()*rawImageMetadata->getHeight();
		}

		if (mRoi.width)
		{
			if (mRoi.x + mRoi.width > rawImageMetadata->getWidth() || mRoi.y + mRoi.height > rawImageMetadata->getHeight())
			{
				auto msg = "ROI exceeds image bounds x<" + to_string(mRoi.x) + "> y<" + to_string(mRoi.y) + "> w<" + to_string(mRoi.width) + "> h<" + to_string(mRoi.height) + "> iw<" + to_string(rawImageMetadata->getWidth()) + "> ih<" + to_string(rawImageMetadata->getHeight()) + ">";
				throw AIPException(AIP_ROI_OUTOFRANGE, msg);
			}
		}

		if (maskImg.data)
		{
			if (maskImg.rows != rawImageMetadata->getHeight() || maskImg.cols != rawImageMetadata->getWidth() || maskImg.channels() != 1)
			{
				auto msg = string("mask image should exactly match input image. Width Height Channels");
				throw AIPException(AIP_ROI_OUTOFRANGE, msg);
			}
		}
	}

	void setInputImg(frame_sp& frame)
	{
		mInputImg.data = (uchar *)frame->data() + mXOffset;
	}
	
	void setOutputImgData(frame_sp& frame)
	{
		mOutputImg.data = (uchar *)frame->data();
	}

	cv::Rect mRoi;	
	size_t mXOffset;
	cv::Mat mInputImg;
	cv::Mat mOutputImg;
	framemetadata_sp mOutputMetadata;
	framemetadata_sp mInputMetadata;

	cv::Mat maskImg;	
	int channelNumber = 0;
	int numberOfPixels = 0;
	float* range;
	const float** ranges;

	boost::shared_ptr<CalcHistogramCVProps> mProps;
};


CalcHistogramCV::CalcHistogramCV(CalcHistogramCVProps _props):Module(TRANSFORM, "CalcHistogramCV", _props)
{
	mDetail.reset(new Detail(_props));
}

bool CalcHistogramCV::validateInputPins()
{
	if (getNumberOfInputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstInputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::RAW_IMAGE)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be RAW_IMAGE. Actual<" << frameType << ">";
		return false;
	}

	return true;
}

bool CalcHistogramCV::validateOutputPins()
{
	if (getNumberOfOutputPins() > 2)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 2. Actual<" << getNumberOfOutputPins() << ">";
		return false;
	}

	pair<string, framemetadata_sp> me; // map element	
	auto metadataByPin = getOutputMetadata();
	BOOST_FOREACH(me, metadataByPin) {
		FrameMetadata::FrameType frameType = me.second->getFrameType();
		if (frameType != FrameMetadata::RAW_IMAGE && frameType != FrameMetadata::ARRAY)
		{
			LOG_ERROR << "<" << getId() << ">::validateOutputPins input frameType is expected to be RAW_IMAGE OR ARRAY. Actual<" << frameType << ">";
			return false;
		}
	}

	return true;
}

bool CalcHistogramCV::validateInputOutputPins()
{
	if (getNumberOfOutputPins() != 2)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 2. Actual<" << getNumberOfOutputPins() << ">";
		return false;
	}

	return Module::validateInputOutputPins();
}

void CalcHistogramCV::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);
	addOutputPin(metadata, pinId);
}

bool CalcHistogramCV::init()
{
	if (!Module::init())
	{
		return false;
	}

	auto metadata = getInputMetadataByType(FrameMetadata::RAW_IMAGE);
	if (metadata->isSet())
	{
		mDetail->setInputMetadata(metadata);
	}

	metadata = getOutputMetadataByType(FrameMetadata::ARRAY);	
	// setting the datasize
	FrameMetadataFactory::downcast<ArrayMetadata>(metadata)->setData(mDetail->bins, CV_32FC1, sizeof(float));
	mDetail->setOutputMetadata(metadata);

	return true;
}

bool CalcHistogramCV::term()
{
	return Module::term();
}

bool CalcHistogramCV::process(frame_container& frames)
{		
	auto frame = getFrameByType(frames, FrameMetadata::RAW_IMAGE);
	if (isFrameEmpty(frame))
	{
		return true;
	}

	// makeframe
	auto metadata = mDetail->getOutputMetadata();
	auto outFrame = makeFrame(metadata->getDataSize(), metadata);
	
	mDetail->compute(frame, outFrame);	
		
	auto pinId = getOutputPinIdByType(FrameMetadata::ARRAY);
	frames.insert(make_pair(pinId, outFrame));	
	send(frames);

	return true;
}

bool CalcHistogramCV::processSOS(frame_sp& frame)
{
	auto metadata = frame->getMetadata();
	mDetail->setInputMetadata(metadata);
	return true;
}

bool CalcHistogramCV::shouldTriggerSOS()
{
	return mDetail->shouldTriggerSOS();
}

CalcHistogramCVProps CalcHistogramCV::getProps()
{
	auto props = mDetail->getProps();
	fillProps(props);

	return props;
}

void CalcHistogramCV::setProps(CalcHistogramCVProps& props)
{
	Module::addPropsToQueue(props);
}

bool CalcHistogramCV::handlePropsChange(frame_sp& frame)
{
	CalcHistogramCVProps props;
	bool ret = Module::handlePropsChange(frame, props);
		
	auto metadata = getOutputMetadataByType(FrameMetadata::ARRAY);
	// setting the datasize
	FrameMetadataFactory::downcast<ArrayMetadata>(metadata)->setData(props.bins, CV_32FC1, sizeof(float));
	mDetail->setOutputMetadata(metadata);
	
	mDetail->setProps(props);
	
	sendEOS();

	return ret;
}