#include "H264Decoder.h"

#ifdef ARM64
#include "H264DecoderV4L2Helper.h"
#else
#include "H264DecoderNvCodecHelper.h"
#endif 

#include "H264ParserUtils.h"
#include "FrameMetadata.h"
#include "H264Metadata.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "H264Utils.h"
#include <typeinfo>

class H264Decoder::Detail
{
public:
	Detail(H264DecoderProps& _props) : mWidth(0), mHeight(0)
	{
	}

	~Detail()
	{
		helper.reset();
	}

	bool setMetadata(framemetadata_sp& metadata, frame_sp frame, std::function<void(frame_sp&)> send, std::function<frame_sp()> makeFrame)
	{
		auto type = H264Utils::getNALUType((char*)frame->data());
		if (type == H264Utils::H264_NAL_TYPE_IDR_SLICE || type == H264Utils::H264_NAL_TYPE_SEQ_PARAM)
		{
			if (metadata->getFrameType() == FrameMetadata::FrameType::H264_DATA)
			{
				sps_pps_properties p;
				H264ParserUtils::parse_sps(((const char*)frame->data()) + 5, frame->size() > 5 ? frame->size() - 5 : frame->size(), &p);
				mWidth = p.width;
				mHeight = p.height;

				auto h264Metadata = framemetadata_sp(new H264Metadata(mWidth, mHeight));
				auto rawOutMetadata = FrameMetadataFactory::downcast<H264Metadata>(h264Metadata);
				rawOutMetadata->setData(*rawOutMetadata);
#ifdef ARM64
				helper.reset(new h264DecoderV4L2Helper());
				return helper->init(send, makeFrame);
#else
				helper.reset(new H264DecoderNvCodecHelper(mWidth, mHeight));
				return helper->init(send, makeFrame);
#endif
			}

			else
			{
				throw AIPException(AIP_NOTIMPLEMENTED, "Unknown frame type");
			}
		}
		else
		{
			return false;
		}
	}

	void compute(frame_sp& frame)
	{
		helper->process(frame);
	}

#ifdef ARM64
	void closeAllThreads(frame_sp eosFrame)
	{
		helper->closeAllThreads(eosFrame);
	}
#endif
public:
	int mWidth;
	int mHeight;
private:

#ifdef ARM64
	boost::shared_ptr<h264DecoderV4L2Helper> helper;
#else
	boost::shared_ptr<H264DecoderNvCodecHelper> helper;
#endif
};

H264Decoder::H264Decoder(H264DecoderProps _props) : Module(TRANSFORM, "H264Decoder", _props), mShouldTriggerSOS(true), mProps(_props)
{
	mDetail.reset(new Detail(mProps));
#ifdef ARM64
	mOutputMetadata = boost::shared_ptr<FrameMetadata>(new RawImagePlanarMetadata(FrameMetadata::MemType::DMABUF));
#else
	mOutputMetadata = boost::shared_ptr<FrameMetadata>(new RawImagePlanarMetadata(RawImageMetadata::MemType::HOST));
#endif
	mOutputPinId = Module::addOutputPin(mOutputMetadata);
}

H264Decoder::~H264Decoder() {}

bool H264Decoder::validateInputPins()
{
	auto numberOfInputPins = getNumberOfInputPins();
	if (numberOfInputPins != 1 && numberOfInputPins != 2)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1 or 2. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstInputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::FrameType::H264_DATA)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be H264_DATA. Actual<" << frameType << ">";
		return false;
	}

	return true;
}

bool H264Decoder::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstOutputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::RAW_IMAGE_PLANAR)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input frameType is expected to be RAW_IMAGE_PLANAR. Actual<" << frameType << ">";
		return false;
	}

	return true;
}

void H264Decoder::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);
}

bool H264Decoder::init()
{
	if (!Module::init())
	{
		return false;
	}
	return true;
}

bool H264Decoder::term()
{
#ifdef ARM64
	auto eosFrame = frame_sp(new EoSFrame());
	mDetail->closeAllThreads(eosFrame);
#endif
	mDetail.reset();
	return Module::term();
}

void H264Decoder::prependSpsPps(uint8_t* iFrameBuffer)
{
	char NaluSeprator[4] = { 00 ,00, 00 ,01 };
	auto nalu = reinterpret_cast<uint8_t*>(NaluSeprator);
	memcpy(iFrameBuffer, nalu, 4);
	iFrameBuffer += 4;
	memcpy(iFrameBuffer, spsBuffer.data(), spsBuffer.size());
	iFrameBuffer += spsBuffer.size();
	memcpy(iFrameBuffer, nalu, 4);
	iFrameBuffer += 4;
	memcpy(iFrameBuffer, ppsBuffer.data(), ppsBuffer.size());
	iFrameBuffer += ppsBuffer.size();;
}

void H264Decoder::clearIncompleteBwdGopTsFromIncomingTSQ(std::deque<frame_sp>& latestGop)
{
	while (!latestGop.empty())
	{
		auto deleteItr = std::find(incomingFramesTSQ.begin(), incomingFramesTSQ.end(), latestGop.front()->timestamp);
		if (deleteItr != incomingFramesTSQ.end())
		{
			incomingFramesTSQ.erase(deleteItr);
			latestGop.pop_front();
		}
	}
}

void H264Decoder::bufferBackwardEncodedFrames(frame_sp& frame, short naluType)
{
	if (hasDirectionChangedToBackward)
	{
		if (!latestBackwardGop.empty())
		{
			latestBackwardGop.clear();
		}

		hasDirectionChangedToBackward = false;
	}
	// insert frames into the latest gop until I frame comes.
	latestBackwardGop.emplace_back(frame);
	// The latest GOP is complete when I Frame comes up, move the GOP to backwardGopBuffer where all the backward GOP's are buffered
	if (naluType == H264Utils::H264_NAL_TYPE_IDR_SLICE || naluType == H264Utils::H264_NAL_TYPE_SEQ_PARAM)
	{
		foundIFrameOfReverseGop = true;
		backwardGopBuffer.push_back(std::move(latestBackwardGop));
	}
}

void H264Decoder::bufferAndDecodeForwardEncodedFrames(frame_sp& frame, short naluType)
{
	if (hasDirectionChangedToForward)
	{
		//Corner case :Whenever the direction changes to forward we just send all the backward buffered GOP's to decoded in a single shot . The motive is to send the current forward frame to decoder in the same step.
		if (!backwardGopBuffer.empty())
		{
			while (!backwardGopBuffer.empty())
			{
				while (!backwardGopBuffer.front().empty())
				{
					decodeFrameFromBwdGOP();
				}
				backwardGopBuffer.pop_front();
			}
			foundIFrameOfReverseGop = false;
		}
		//Corner case : Whenever direction changes to forward , And the latestBackwardGop is incomplete , then delete the latest backward GOP and remove the frames from incomingFramesTSQ entry as well
		if (!latestBackwardGop.empty())
		{
			clearIncompleteBwdGopTsFromIncomingTSQ(latestBackwardGop);
		}

		if (!latestForwardGop.empty())
		{
			short naluTypeOfForwardGopFirstFrame = H264Utils::getNALUType((char*)latestForwardGop.front()->data());
			if (naluTypeOfForwardGopFirstFrame == H264Utils::H264_NAL_TYPE_IDR_SLICE || naluTypeOfForwardGopFirstFrame == H264Utils::H264_NAL_TYPE_SEQ_PARAM)
			{
				// Corner case: Forward :- When end of cache in the middle of GOP , 
				if (latestForwardGop.front()->timestamp > frame->timestamp)
				{
					latestForwardGop.clear();
				}
			}

			// Corner case: Forward:- When end of cache hits while in the middle of gop, before decoding the next P frame we need decode the previous frames of that GOP. 
			// There might be a case where we might have cleared the decoder by sending empty frame , in order to start the decoder again we must prepend sps and pps to I frame if not present
			if (!latestForwardGop.empty() && naluTypeOfForwardGopFirstFrame == H264Utils::H264_NAL_TYPE_IDR_SLICE)
			{
				auto iFrame = makeFrame(latestForwardGop.front()->size() + spsBuffer.size() + ppsBuffer.size() + 8);
				uint8_t* tempFrameBuffer = reinterpret_cast<uint8_t*>(iFrame->data());
				prependSpsPps(tempFrameBuffer);
				tempFrameBuffer += spsBuffer.size() + ppsBuffer.size() + 8;
				memcpy(tempFrameBuffer, latestForwardGop.front()->data(), latestForwardGop.front()->size());
				iFrame->timestamp = latestForwardGop.front()->timestamp;
				mDetail->compute(iFrame);
				latestForwardGop.pop_front();
				for (auto itr = latestForwardGop.begin(); itr != latestForwardGop.end(); itr++)
				{
					if (itr->get()->timestamp < frame->timestamp)
					{
						mDetail->compute(*itr);
					}
				}
			}
			else if (!latestForwardGop.empty() && naluTypeOfForwardGopFirstFrame == H264Utils::H264_NAL_TYPE_SEQ_PARAM)
			{
				for (auto itr = latestForwardGop.begin(); itr != latestForwardGop.end(); itr++)
				{
					if (itr->get()->timestamp < frame->timestamp)
					{
						mDetail->compute(*itr);
					}
				}
			}
		}
		hasDirectionChangedToForward = false;
	}
	cacheResumedInForward = false;
	/* buffer fwd GOP and send the current frame */

	// new GOP starts
	if (naluType == H264Utils::H264_NAL_TYPE_IDR_SLICE || naluType == H264Utils::H264_NAL_TYPE_SEQ_PARAM)
	{
		latestForwardGop.clear();
	}
	latestForwardGop.emplace_back(frame);
	short latestForwardGopFirstFrameNaluType = H264Utils::getNALUType((char*)latestForwardGop.begin()->get()->data());
	// corner case: Forward: If direction changed to forward in the middle of GOP (Even the latest gop of backward was half and not decoded) , Then we drop the P frames until I frame.
	//We also remove the entries of P frames from the incomingFramesTSQ.
	if (latestForwardGopFirstFrameNaluType != H264Utils::H264_NAL_TYPE_IDR_SLICE && latestForwardGopFirstFrameNaluType != H264Utils::H264_NAL_TYPE_SEQ_PARAM)
	{
		clearIncompleteBwdGopTsFromIncomingTSQ(latestForwardGop);
		return;
	}

	mDetail->compute(frame);//dont send sps frame when direction changes - todo
	return;
}

void H264Decoder::decodeFrameFromBwdGOP()
{
	if (!backwardGopBuffer.empty() && !backwardGopBuffer.front().empty())
	{
		// For reverse play we sent the frames to the decoder in reverse, As the last frame added in the deque should be sent first (Example : P,P,P,P,P,P,I)
		auto itr = backwardGopBuffer.front().rbegin();
		mDetail->compute(*itr);
		backwardGopBuffer.front().pop_back();
		return;
	}
}

void H264Decoder::saveSpsPps(frame_sp frame)
{
	auto mFrameBuffer = const_buffer(frame->data(), frame->size());
	auto ret = H264Utils::parseNalu(mFrameBuffer);
	const_buffer tempSpsBuffer;
	const_buffer tempPpsBuffer;
	short typeFound;
	tie(typeFound, tempSpsBuffer, tempPpsBuffer) = ret;

	if ((tempSpsBuffer.size() != 0) || (tempPpsBuffer.size() != 0))
	{
		mHeaderFrame = frame;
		spsBuffer = tempSpsBuffer;
		ppsBuffer = tempPpsBuffer;
	}
}

bool H264Decoder::process(frame_container& frames)
{
	auto frame = frames.begin()->second;
	auto frameMetadata = frame->getMetadata();
	auto h264Metadata = FrameMetadataFactory::downcast<H264Metadata>(frameMetadata);

	if (mDirection && !h264Metadata->direction)
	{
		hasDirectionChangedToBackward = true;
	}
	else if (!mDirection && h264Metadata->direction)
	{
		hasDirectionChangedToForward = true;
	}
	else
	{
		hasDirectionChangedToBackward = false;
		hasDirectionChangedToForward = false;
	}

	/* Clear the latest forward gop whenever seek happens bcz there is no buffering for fwd play.
	We dont clear backwardGOP because there might be a left over GOP to be decoded. */
	if (h264Metadata->mp4Seek)
	{
		latestForwardGop.clear();
	}

	mDirection = h264Metadata->direction;
	short naluType = H264Utils::getNALUType((char*)frame->data());
	if (naluType == H264Utils::H264_NAL_TYPE_IDR_SLICE || naluType == H264Utils::H264_NAL_TYPE_SEQ_PARAM)
	{
		saveSpsPps(frame);
	}
	// we get a repeated frame whenever direction changes i.e. the timestamp Q latest frame is repeated
	if (!incomingFramesTSQ.empty() && incomingFramesTSQ.back() == frame->timestamp)
	{
		flushDecoderFlag = true;
	}

	//Insert the frames time stamp in TS queue. We send the frames to next modules in the same order.
	incomingFramesTSQ.push_back(frame->timestamp);

	//If the frame is already present in the decoded output queue then skip the frame decoding.
	if (decodedFramesCache.find(frame->timestamp) != decodedFramesCache.end())
	{
		// the buffered GOPs in bwdGOPBuffer needs to need to be processed first
		while (!backwardGopBuffer.empty())
		{
			while (!backwardGopBuffer.front().empty())
			{
				decodeFrameFromBwdGOP();
			}
			// remove the decoded GOP now
			backwardGopBuffer.pop_front();
		}

		/* todo: test this logic again */

		// if we seeked
		if (h264Metadata->mp4Seek)
		{
			// flush the incomplete GOP
			flushDecoderFlag = true;
			clearIncompleteBwdGopTsFromIncomingTSQ(latestBackwardGop);
		}

		// GOPs processed, reset flag
		foundIFrameOfReverseGop = false;

		// corner case: partial GOP already present in cache 
		// todo: double check this if condition
		if (latestBackwardGop.empty())
		{
			flushDecoderFlag = true;
		}

		if (!latestBackwardGop.empty())
		{
			// Corner case: backward :- (I,P,P,P) Here if first two frames are in the cache and last two frames are not in the cache , to decode the last two frames we buffer the full gop and later decode it.
			bufferBackwardEncodedFrames(frame, naluType);
			sendDecodedFrame();
			return true;
		}

		//Corner case: Forward:- Special buffering, Starting some frames of current GOP can be there in the decoded cache and later frames may not be there in the decoded cache,
		// to decoded the later frames we always buffer the current GOP
		// check if a new fwd GOP has started
		if (!latestForwardGop.empty() && h264Metadata->direction && ((naluType == H264Utils::H264_NAL_TYPE_SEQ_PARAM) || (naluType == H264Utils::H264_NAL_TYPE_IDR_SLICE)))
		{
			latestForwardGop.clear();
		}

		if (h264Metadata->direction) // todo: verify and add/delete && (latestForwardGop.empty()) && (H264Utils::getNALUType((char*)latestForwardGop.front()->data()) == H264Utils::H264_NAL_TYPE_SEQ_PARAM) || (H264Utils::getNALUType((char*)latestForwardGop.front()->data()) == H264Utils::H264_NAL_TYPE_IDR_SLICE)
		{
			flushDecoderFlag = false;
			latestForwardGop.push_back(frame);
			hasDirectionChangedToForward = true; // Corner case: Its a small hack - Only starting frames of GOP are in the cache , in order to decode the later P frames we set this flag to true , where in first the latestForwardGop is decoded and later the current P frame
		}

		//Corner case : forward:- While in forward play, if cache has resumed in the middle of the GOP then to get the previous few frames we need to flush the decoder.
		if (h264Metadata->direction && !cacheResumedInForward)
		{
			auto eosFrame = frame_sp(new EmptyFrame());
			mDetail->compute(eosFrame);
			cacheResumedInForward = true;
		}
		sendDecodedFrame();
		return true;
	}

	/* If frame is not in output cache, it needs to be buffered & decoded */
	if (mDirection)
	{
		//Buffers the latest GOP and send the current frame to decoder.
		bufferAndDecodeForwardEncodedFrames(frame, naluType);
	}
	else
	{
		//Only buffering of backward GOP happens 
		bufferBackwardEncodedFrames(frame, naluType);
	}
	if (foundIFrameOfReverseGop)
	{
		// The I frame of backward GOP was found , now we send the frames to the decoder one by one in every step
		decodeFrameFromBwdGOP();
		if (backwardGopBuffer.size() == 1 && backwardGopBuffer.front().empty())
		{
			backwardGopBuffer.pop_front();
			foundIFrameOfReverseGop = false;
		}
	}
	sendDecodedFrame();
	dropFarthestFromCurrentTs(frame->timestamp);
	return true;
}

void H264Decoder::sendDecodedFrame()
{
	// not in output cache && flushdecoder flag is set
	if (!incomingFramesTSQ.empty() && !decodedFramesCache.empty() && decodedFramesCache.find(incomingFramesTSQ.front()) == decodedFramesCache.end() && flushDecoderFlag)
	{
		// We send empty frame to the decoder , in order to flush out all the frames from decoder.
		// This is to handle some cases whenever the direction change happens and to get out the latest few frames sent to decoder.
		auto eosFrame = frame_sp(new EmptyFrame());
		mDetail->compute(eosFrame);
		flushDecoderFlag = false;
		latestBackwardGop.clear();
	}

	// timestamp in output cache
	if (!incomingFramesTSQ.empty() && !decodedFramesCache.empty() && decodedFramesCache.find(incomingFramesTSQ.front()) != decodedFramesCache.end())
	{
		auto outFrame = decodedFramesCache[incomingFramesTSQ.front()];
		incomingFramesTSQ.pop_front();
		frame_container frames;
		frames.insert(make_pair(mOutputPinId, outFrame));
		send(frames);
	}
}

void H264Decoder::bufferDecodedFrames(frame_sp& frame)
{
	decodedFramesCache.insert({ frame->timestamp, frame });
}

void H264Decoder::dropFarthestFromCurrentTs(uint64_t ts)
{
	if (decodedFramesCache.empty())
	{
		return;
	}

	/* dropping algo */
	int64_t begDistTS = ts - decodedFramesCache.begin()->first;
	auto absBeginDistance = abs(begDistTS);
	int64_t endDistTS = ts - decodedFramesCache.rbegin()->first;
	auto absEndDistance = abs(endDistTS);
	if (decodedFramesCache.size() >= mProps.upperWaterMark)
	{
		if (absEndDistance <= absBeginDistance)
		{
			auto itr = decodedFramesCache.begin();
			while (itr != decodedFramesCache.end())
			{
				if (decodedFramesCache.size() >= mProps.lowerWaterMark)
				{
					boost::mutex::scoped_lock(m_mutex);
					// Note - erase returns the iterator of next element after deletion.
					// Dont drop the frames from cache which are present in the incomingFramesTSQ
					if (std::find(incomingFramesTSQ.begin(), incomingFramesTSQ.end(), itr->first) != incomingFramesTSQ.end())
					{
						itr++;
						continue;
					}
					itr = decodedFramesCache.erase(itr);
				}
				else
				{
					return;
				}
			}
		}
		else
		{
			// delete from end using the fwd iterator.
			auto itr = decodedFramesCache.end();
			--itr;
			while (itr != decodedFramesCache.begin())
			{
				if (decodedFramesCache.size() >= mProps.lowerWaterMark)
				{
					boost::mutex::scoped_lock(m_mutex);
					// Note - erase returns the iterator of next element after deletion.
					if (std::find(incomingFramesTSQ.begin(), incomingFramesTSQ.end(), itr->first) != incomingFramesTSQ.end())
					{
						--itr;
						continue;
					}
					itr = decodedFramesCache.erase(itr);
					--itr;
				}
				else
				{
					return;
				}
			}
		}
	}
}

bool H264Decoder::processSOS(frame_sp& frame)
{
	auto metadata = frame->getMetadata();
	auto h264Metadata = FrameMetadataFactory::downcast<H264Metadata>(metadata);
	mDirection = h264Metadata->direction;
	auto ret = mDetail->setMetadata(metadata, frame,
		[&](frame_sp& outputFrame) {
			/*auto bDF = std::thread(&H264Decoder::bufferDecodedFrames, this, outputFrame);
			bDF.join();*/
			bufferDecodedFrames(outputFrame);
		}, [&]() -> frame_sp {return makeFrame(); }
		);
	if (ret)
	{
		mShouldTriggerSOS = false;
		auto rawOutMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(mOutputMetadata);

#ifdef ARM64
		RawImagePlanarMetadata OutputMetadata(mDetail->mWidth, mDetail->mHeight, ImageMetadata::ImageType::NV12, 128, CV_8U, FrameMetadata::MemType::DMABUF);
#else
		RawImagePlanarMetadata OutputMetadata(mDetail->mWidth, mDetail->mHeight, ImageMetadata::YUV420, size_t(0), CV_8U, FrameMetadata::HOST);
#endif

		rawOutMetadata->setData(OutputMetadata);
	}

	return true;
}

bool H264Decoder::shouldTriggerSOS()
{
	return mShouldTriggerSOS;
}

bool H264Decoder::processEOS(string& pinId)
{
	auto frame = frame_sp(new EmptyFrame());
	mDetail->compute(frame);
	mShouldTriggerSOS = true;
	return true;
}

//Todo : override flusQue method here