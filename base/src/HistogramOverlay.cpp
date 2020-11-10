#include <opencv2/highgui.hpp>
#include <boost/foreach.hpp>
#include <boost/math/special_functions/round.hpp>

#include "HistogramOverlay.h"
#include "Frame.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Utils.h"
#include "Logger.h"
#include "AIPExceptions.h"

class HistogramOverlay::Detail
{

public:
	Detail(): mSpacing(0) {}

	~Detail() {}
		
	void setImgMetadata(framemetadata_sp& metadata)
	{		
		mOutputMetadata = metadata;
		mOutputImg = Utils::getMatHeader(FrameMetadataFactory::downcast<RawImageMetadata>(metadata));		
	}

	void resetHistMetadata()
	{
		mHistMetadata.reset();
		mSpacing = 0;
		contours.clear();
	}

	void setHistMetadata(framemetadata_sp& metadata)
	{		
		auto arrayMetadata = FrameMetadataFactory::downcast<ArrayMetadata>(metadata);
		mHist = Utils::getMatHeader(1, arrayMetadata->getLength(), arrayMetadata->getType());
		mHistMetadata = metadata;
	}

	void overlayHistogram(frame_sp& imgFrame, frame_sp& histFrame)
	{
		mHist.data = (uchar *)histFrame->data();		
		mOutputImg.data = (uchar *)imgFrame->data();

		double maxVal = 0;
		cv::minMaxLoc(mHist, 0, &maxVal, 0, 0);

		auto height = mOutputImg.rows;
		auto maxHeight = mOutputImg.rows/2;
		if (mSpacing == 0)
		{
			// spacing not set
			setSpacing();
		}

		vector<cv::Point>& contour = contours[0];						
		for (int h = 0; h < mHist.rows; h++)
		{
			float binVal = mHist.at<float>(h, 0);
			int ypos = height;
			if (maxVal != 0)
			{
				ypos = height - cvRound(binVal * maxHeight / maxVal);
			}
			contour[h+1].y = ypos;
		}

		cv::drawContours(mOutputImg, contours, 0, cv::Scalar(255, 0, 0), cv::LINE_4, 8);
	}

	void setSpacing(int spacing = 0)
	{
		auto bins = mHist.rows;					
		mSpacing = spacing;
		if (mSpacing == 0)
		{
			mSpacing = 100; // initial value
		}

		if (mSpacing*(bins+2) > 0.75*mOutputImg.cols)
		{
			return setSpacing(boost::math::iround(mSpacing*0.75));
		}

		if (mSpacing < 20)
		{
			auto msg = "Not Enough width to overlay histogram of size<" + to_string(bins) + ">";
			throw AIPException(AIP_PARAM_OUTOFRANGE, msg);
		}

		auto maxHeight = mOutputImg.rows / 2;
		if (maxHeight < 30)
		{
			auto msg = "Not Enough height to overlay histogram.";
			throw AIPException(AIP_PARAM_OUTOFRANGE, string(msg));
		}

		vector<cv::Point> contour;
		contour.push_back(cv::Point(mSpacing*(2), mOutputImg.rows));
		for (int h = 0; h < mHist.rows; h++)
		{
			contour.push_back(cv::Point(mSpacing*(h + 2), 0));
		}
		contour.push_back(cv::Point(mSpacing*(mHist.rows+1), mOutputImg.rows));

		contours.push_back(contour);
	}	
	
	framemetadata_sp getOutputMetadata()
	{
		return mOutputMetadata;
	}	

	bool shouldTriggerSOS()
	{
		return !mHistMetadata.get() || !mOutputMetadata.get() || !mHist.rows;
	}
	
private:		
	cv::Mat mHist;
	cv::Mat mOutputImg;
	framemetadata_sp mOutputMetadata;	
	framemetadata_sp mHistMetadata;
	vector<vector<cv::Point> > contours;
	int mSpacing;
};


HistogramOverlay::HistogramOverlay(HistogramOverlayProps _props):Module(TRANSFORM, "HistogramOverlay", _props)
{
	mDetail.reset(new Detail());
}

bool HistogramOverlay::validateInputPins()
{
	if (getNumberOfInputPins() > 2)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 2. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	pair<string, framemetadata_sp> me; // map element	
	auto inputMetadataByPin = getInputMetadata();
	BOOST_FOREACH(me, inputMetadataByPin) {
		FrameMetadata::FrameType frameType = me.second->getFrameType();		
		if (frameType != FrameMetadata::RAW_IMAGE && frameType != FrameMetadata::ARRAY)
		{
			LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be RAW_IMAGE OR ARRAY. Actual<" << frameType << ">";
			return false;
		}
	}

	return true;
}

bool HistogramOverlay::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstOutputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::RAW_IMAGE)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input frameType is expected to be RAW_IMAGE. Actual<" << frameType << ">";
		return false;
	}

	return true;
}

bool HistogramOverlay::validateInputOutputPins()
{
	if (getNumberOfInputPins() != 2)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 2. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}
	
	return Module::validateInputOutputPins();
}

bool HistogramOverlay::init()
{
	if (!Module::init())
	{
		return false;
	}

	auto metadata = getInputMetadataByType(FrameMetadata::RAW_IMAGE);
	if (metadata->isSet())
	{
		mDetail->setImgMetadata(metadata);
	}
	metadata = getInputMetadataByType(FrameMetadata::ARRAY);
	if (metadata->isSet())
	{
		mDetail->setHistMetadata(metadata);
	}
	
	return true;
}

bool HistogramOverlay::term()
{
	return Module::term();
}

bool HistogramOverlay::process(frame_container& frames)
{		
	auto imgFrame = getFrameByType(frames, FrameMetadata::RAW_IMAGE);
	auto histFrame = getFrameByType(frames, FrameMetadata::ARRAY);

	if (!imgFrame.get() || !histFrame.get())
	{
		LOG_INFO << "Image and histogram both are required. So skipping.<" << getId() << ">";
		return true;
	}

	// makeframe
	auto metadata = mDetail->getOutputMetadata();
	auto outFrame = makeFrame(metadata->getDataSize(), metadata);
		
	// copy buffer
	memcpy(outFrame->data(), imgFrame->data(), metadata->getDataSize());

	mDetail->overlayHistogram(outFrame, histFrame);

	frames.insert(make_pair(getOutputPinIdByType(FrameMetadata::RAW_IMAGE), outFrame));
	send(frames);

	return true;
}

bool HistogramOverlay::processSOS(frame_sp& frame)
{
	auto metadata = frame->getMetadata();
	if (metadata->getFrameType() == FrameMetadata::RAW_IMAGE)
	{
		mDetail->setImgMetadata(metadata);
	}
	else if (metadata->getFrameType() == FrameMetadata::ARRAY)
	{
		mDetail->setHistMetadata(metadata);
	}

	return true;
}

bool HistogramOverlay::shouldTriggerSOS()
{
	return mDetail->shouldTriggerSOS();
}

bool HistogramOverlay::processEOS(string& pinId)
{
	mDetail->resetHistMetadata();
	return true;
}