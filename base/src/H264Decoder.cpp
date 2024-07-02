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

	void compute(void* inputFrameBuffer, size_t inputFrameSize, uint64_t inputFrameTS)
	{
		if(helper != nullptr)
		{
			helper->process(inputFrameBuffer, inputFrameSize, inputFrameTS);
		}
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

void* H264Decoder::prependSpsPps(frame_sp& iFrame, size_t& spsPpsFrameSize)
{
	spsPpsFrameSize = iFrame->size() + spsBuffer.size() + ppsBuffer.size() + 8;
	uint8_t* spsPpsFrameBuffer = new uint8_t[spsPpsFrameSize];
	char NaluSeprator[4] = { 00 ,00, 00 ,01 };
	auto nalu = reinterpret_cast<uint8_t*>(NaluSeprator);
	memcpy(spsPpsFrameBuffer, nalu, 4);
	spsPpsFrameBuffer += 4;
	memcpy(spsPpsFrameBuffer, spsBuffer.data(), spsBuffer.size());
	spsPpsFrameBuffer += spsBuffer.size();
	memcpy(spsPpsFrameBuffer, nalu, 4);
	spsPpsFrameBuffer += 4;
	memcpy(spsPpsFrameBuffer, ppsBuffer.data(), ppsBuffer.size());
	spsPpsFrameBuffer += ppsBuffer.size();
	memcpy(spsPpsFrameBuffer, iFrame->data(), iFrame->size());
	spsPpsFrameBuffer = spsPpsFrameBuffer - spsBuffer.size() - ppsBuffer.size() - 8;
	return spsPpsFrameBuffer;
}

void H264Decoder::clearIncompleteBwdGopTsFromIncomingTSQ(std::deque<frame_sp>& latestGop)
{
	while (!latestGop.empty() && !incomingFramesTSQ.empty())
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
	if (dirChangedToBwd)
	{
		latestBackwardGop.clear();
		dirChangedToBwd = false;
	}
	// insert frames into the latest gop until I frame comes.
	latestBackwardGop.emplace_back(frame);
	H264Utils::H264_NAL_TYPE nalTypeAfterSpsPps = (H264Utils::H264_NAL_TYPE)1;
	if(naluType == H264Utils::H264_NAL_TYPE_SEQ_PARAM)
	{
		nalTypeAfterSpsPps = H264Utils::getNalTypeAfterSpsPps(frame->data(), frame->size());
	}
	// The latest GOP is complete when I Frame comes up, move the GOP to backwardGopBuffer where all the backward GOP's are buffered
	if (naluType == H264Utils::H264_NAL_TYPE_IDR_SLICE || nalTypeAfterSpsPps == H264Utils::H264_NAL_TYPE_IDR_SLICE)
	{
		foundIFrameOfReverseGop = true;
		backwardGopBuffer.push_back(std::move(latestBackwardGop));
	}
}

void H264Decoder::bufferAndDecodeForwardEncodedFrames(frame_sp& frame, short naluType)
{
	if (dirChangedToFwd)
	{
		// Whenever the direction changes to forward we just send all the backward buffered GOP's to decoded in a single step . The motive is to send the current forward frame to decoder in the same step.
		while (!backwardGopBuffer.empty())
		{
			decodeFrameFromBwdGOP();
		}

		// Whenever direction changes to forward , And the latestBackwardGop is incomplete , then delete the latest backward GOP and remove the frames from incomingFramesTSQ entry as well
		if (!latestBackwardGop.empty())
		{
			clearIncompleteBwdGopTsFromIncomingTSQ(latestBackwardGop);
		}
		dirChangedToFwd = false;
	}
	if(prevFrameInCache)
	{
		// previous Frame was In Cache & current is not
		if (!latestForwardGop.empty())
		{
			short naluTypeOfForwardGopFirstFrame = H264Utils::getNALUType((char*)latestForwardGop.front()->data());
			H264Utils::H264_NAL_TYPE nalTypeAfterSpsPpsOfForwardGopFirstFrame = (H264Utils::H264_NAL_TYPE)1;
			if(naluTypeOfForwardGopFirstFrame == H264Utils::H264_NAL_TYPE_SEQ_PARAM)
			{
				nalTypeAfterSpsPpsOfForwardGopFirstFrame = H264Utils::getNalTypeAfterSpsPps(latestForwardGop.front()->data(), latestForwardGop.front()->size());
			}
			if (naluTypeOfForwardGopFirstFrame == H264Utils::H264_NAL_TYPE_IDR_SLICE || nalTypeAfterSpsPpsOfForwardGopFirstFrame == H264Utils::H264_NAL_TYPE_IDR_SLICE)
			{
				// Corner case: Forward :- current frame is not part of latestForwardGOP 
				if (latestForwardGop.front()->timestamp > frame->timestamp)
				{
					latestForwardGop.clear();
				}
			}

			// Corner case: Forward:- When end of cache hits while in the middle of gop, before decoding the next P frame we need decode the previous frames of that GOP. 
			// There might be a case where we might have cleared the decoder, in order to start the decoder again we must prepend sps and pps to I frame if not present
			if (!latestForwardGop.empty() && naluTypeOfForwardGopFirstFrame == H264Utils::H264_NAL_TYPE_IDR_SLICE)
			{
				auto iFrame = latestForwardGop.front();
				size_t spsPpsFrameSize;
				auto spsPpsFrameBuffer = prependSpsPps(iFrame, spsPpsFrameSize);
				mDetail->compute(spsPpsFrameBuffer, spsPpsFrameSize, iFrame->timestamp);
				latestForwardGop.pop_front();
				for (auto itr = latestForwardGop.begin(); itr != latestForwardGop.end(); itr++)
				{
					if (!latestForwardGop.empty() && itr != latestForwardGop.end() && itr->get()->timestamp < frame->timestamp)
					{
						mDetail->compute(itr->get()->data(), itr->get()->size(), itr->get()->timestamp);
					}
				}
			}
			else if (!latestForwardGop.empty() && nalTypeAfterSpsPpsOfForwardGopFirstFrame == H264Utils::H264_NAL_TYPE_IDR_SLICE)
			{
				for (auto itr = latestForwardGop.begin(); itr != latestForwardGop.end(); itr++)
				{
					if (!latestForwardGop.empty() && itr != latestForwardGop.end() && itr->get()->timestamp < frame->timestamp)
					{
						mDetail->compute(itr->get()->data(), itr->get()->size(), itr->get()->timestamp);
					}
				}
			}
		}
	}
	prevFrameInCache = false;

	/* buffer fwd GOP and send the current frame */
	// new GOP starts
	H264Utils::H264_NAL_TYPE nalTypeAfterSpsPpsOfCurrentFrame = (H264Utils::H264_NAL_TYPE)1;
	if(naluType == H264Utils::H264_NAL_TYPE_SEQ_PARAM)
	{
		nalTypeAfterSpsPpsOfCurrentFrame = H264Utils::getNalTypeAfterSpsPps(frame->data(), frame->size());
	}
	if (naluType == H264Utils::H264_NAL_TYPE_IDR_SLICE || nalTypeAfterSpsPpsOfCurrentFrame == H264Utils::H264_NAL_TYPE_IDR_SLICE)
	{
		latestForwardGop.clear();
	}
	
	latestForwardGop.emplace_back(frame);
	

	// If direction changed to forward in the middle of GOP (Even the latest gop of backward was half and not decoded) , Then we drop the P frames until next I frame.
	// We also remove the entries of P frames from the incomingFramesTSQ.
	short latestForwardGopFirstFrameNaluType = H264Utils::getNALUType((char*)latestForwardGop.begin()->get()->data());
	H264Utils::H264_NAL_TYPE naluTypeAfterSpsPpsOfLatestForwardGopFirstFrame = (H264Utils::H264_NAL_TYPE)1;
	if(latestForwardGopFirstFrameNaluType == H264Utils::H264_NAL_TYPE_SEQ_PARAM)
	{
			naluTypeAfterSpsPpsOfLatestForwardGopFirstFrame = H264Utils::getNalTypeAfterSpsPps(latestForwardGop.front()->data(), latestForwardGop.front()->size());
	}
	if (latestForwardGopFirstFrameNaluType != H264Utils::H264_NAL_TYPE_IDR_SLICE && naluTypeAfterSpsPpsOfLatestForwardGopFirstFrame != H264Utils::H264_NAL_TYPE_IDR_SLICE)
	{
		clearIncompleteBwdGopTsFromIncomingTSQ(latestForwardGop);
		return;
	}

	mDetail->compute(frame->data(), frame->size(), frame->timestamp);
	return;
}

void H264Decoder::decodeFrameFromBwdGOP()
{
	if (!backwardGopBuffer.empty() && !backwardGopBuffer.front().empty() && H264Utils::getNALUType((char*)backwardGopBuffer.front().back()->data()) == H264Utils::H264_NAL_TYPE_IDR_SLICE && prevFrameInCache)
	{
		auto iFrame = backwardGopBuffer.front().back();
		size_t spsPpsFrameSize;
		auto spsPpsFrameBuffer = prependSpsPps(iFrame, spsPpsFrameSize);
		mDetail->compute(spsPpsFrameBuffer, spsPpsFrameSize, iFrame->timestamp);
		
		backwardGopBuffer.front().pop_back();
		
		prevFrameInCache = false;
	}
	
	if (!backwardGopBuffer.empty() && !backwardGopBuffer.front().empty())
	{
		// For reverse play we sent the frames to the decoder in reverse, As the last frame added in the deque should be sent first (Example : P,P,P,P,P,P,I)
		auto itr = backwardGopBuffer.front().rbegin();
		mDetail->compute(itr->get()->data(), itr->get()->size(), itr->get()->timestamp);
		backwardGopBuffer.front().pop_back();
	}
	if (backwardGopBuffer.size() >= 1 && backwardGopBuffer.front().empty())
	{
		backwardGopBuffer.pop_front();
	}
	
	if (backwardGopBuffer.empty())
	{
		foundIFrameOfReverseGop = false;
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
	if(incomingFramesTSQ.size() >= 1000)
	{
		flushQue();
	}
	auto frame = frames.begin()->second;
	auto myId = Module::getId();
	auto frameMetadata = frame->getMetadata();
	auto h264Metadata = FrameMetadataFactory::downcast<H264Metadata>(frameMetadata);

	if (mDirection && !h264Metadata->direction)
	{
		dirChangedToBwd = true;
		resumeBwdPlayback = false;
		LOG_INFO<<"Pausing decoder";

	}
	else if (!mDirection && h264Metadata->direction)
	{
		dirChangedToFwd = true;
		resumeFwdPlayback = false;
		LOG_INFO<<"Pausing decoder";
	}
	else
	{
		dirChangedToBwd = false;
		dirChangedToFwd = false;
	}
	/* Clear the latest forward gop whenever seek happens bcz there is no buffering for fwd play.
	We dont clear backwardGOP because there might be a left over GOP to be decoded. */
	if (h264Metadata->mp4Seek)
	{
		
		latestForwardGop.clear();
		
	}

	mDirection = h264Metadata->direction;
	short naluType = H264Utils::getNALUType((char*)frame->data());

	if ((playbackSpeed == 8 || playbackSpeed == 16 || playbackSpeed == 32) && (naluType != H264Utils::H264_NAL_TYPE_IDR_SLICE && naluType != H264Utils::H264_NAL_TYPE_SEQ_PARAM))
	{
		if ((currentFps * playbackSpeed) / gop > currentFps)
		{
			if (iFramesToSkip)
			{
				iFramesToSkip--;
				return true;
			}
			if (!iFramesToSkip)
			{
				iFramesToSkip = ((currentFps * playbackSpeed) / gop) / currentFps;
			}
		}
	}

	if (naluType == H264Utils::H264_NAL_TYPE_SEQ_PARAM)
	{
		saveSpsPps(frame);
	}

	//Insert the frames time stamp in TS queue. We send the frames to next modules in the same order.
	incomingFramesTSQ.push_back(frame->timestamp);
	//If the frame is already present in the decoded output cache then skip the frame decoding.
	if (decodedFramesCache.find(frame->timestamp) != decodedFramesCache.end())
	{
		//prepend sps and pps if 1st frame is I frame
		if (!backwardGopBuffer.empty() && H264Utils::getNALUType((char*)backwardGopBuffer.front().back()->data()) == H264Utils::H264_NAL_TYPE_IDR_SLICE)
		{
			
			auto iFrame = backwardGopBuffer.front().back();
			
			size_t spsPpsFrameSize;
			auto spsPpsFrameBuffer = prependSpsPps(iFrame, spsPpsFrameSize);
			mDetail->compute(spsPpsFrameBuffer, spsPpsFrameSize, iFrame->timestamp);
			
			backwardGopBuffer.front().pop_back();
			
		}
		// the buffered GOPs in bwdGOPBuffer needs to need to be processed first
		while (!backwardGopBuffer.empty())
		{
			decodeFrameFromBwdGOP();
		}

		// if we seeked
		if (h264Metadata->mp4Seek)
		{
			clearIncompleteBwdGopTsFromIncomingTSQ(latestBackwardGop);
		}

		if (!latestBackwardGop.empty())
		{
			// Corner case: backward :- (I,P,P,P) Here if first two frames are in the cache and last two frames are not in the cache , to decode the last two frames we buffer the full gop and later decode it.
			bufferBackwardEncodedFrames(frame, naluType);
			sendDecodedFrame();
			return true;
		}

		H264Utils::H264_NAL_TYPE nalTypeAfterSpsPpsCurrentFrame = (H264Utils::H264_NAL_TYPE)1;
		if(naluType == H264Utils::H264_NAL_TYPE_SEQ_PARAM)
		{
			nalTypeAfterSpsPpsCurrentFrame = H264Utils::getNalTypeAfterSpsPps(frame->data(), frame->size());
		}
		if (mDirection && ((nalTypeAfterSpsPpsCurrentFrame == H264Utils::H264_NAL_TYPE_IDR_SLICE) || (naluType == H264Utils::H264_NAL_TYPE_IDR_SLICE)))
		{
			latestForwardGop.clear();
			latestForwardGop.push_back(frame);
			
		}
		// dont buffer fwd GOP if I frame has not been recieved (possible in intra GOP direction change cases)
		else if (mDirection && !latestForwardGop.empty() && (nalTypeAfterSpsPpsCurrentFrame == H264Utils::H264_NAL_TYPE_IDR_SLICE || H264Utils::getNALUType((char*)latestForwardGop.front()->data()) == H264Utils::H264_NAL_TYPE_IDR_SLICE))
		{
			latestForwardGop.push_back(frame);
		}

		prevFrameInCache = true;
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
	}
	sendDecodedFrame();
	dropFarthestFromCurrentTs(frame->timestamp);
	return true;
}

void H264Decoder::sendDecodedFrame()
{
	// timestamp in output cache
	if (!incomingFramesTSQ.empty() && !decodedFramesCache.empty() && decodedFramesCache.find(incomingFramesTSQ.front()) != decodedFramesCache.end())
	{
		auto outFrame = decodedFramesCache[incomingFramesTSQ.front()];
		incomingFramesTSQ.pop_front();
		if (resumeBwdPlayback == false && outFrame->timestamp <= lastFrameSent)
		{
			LOG_INFO << "resuming decoder";
			resumeBwdPlayback = true;
		}

		if (resumeFwdPlayback == false && outFrame->timestamp >= lastFrameSent)
		{
			LOG_INFO << "resuming decoder";
			resumeFwdPlayback = true;
		}

		// while (!decodedFramesCache.empty() && incomingFramesTSQSize >0 && resumePlayback == false){
		// 	//take decoder frames and keep popping till incomingFramesTSQSize becomes zero
			
		// 	incomingFramesTSQ.pop_front();
		// 	incomingFramesTSQSize -= 1;
		// }
		// if (incomingFramesTSQSize == 0){
		// 	resumePlayback = true;	
		// 	incomingFramesTSQSize = -1;
		// 	LOG_ERROR<<"Decoder seek playback continues";
		// }
		
		if (incomingFramesTSQSize > 0){
			incomingFramesTSQSize -= 1;
		}
		else if (incomingFramesTSQSize == 0){
			resumePlayback = true;
			LOG_INFO<<"resuming decoder playback ";
			incomingFramesTSQSize = -1;
		}

		if(!framesToSkip)
		{
			frame_container frames;
			frames.insert(make_pair(mOutputPinId, outFrame));
			if(resumePlayback && resumeFwdPlayback && resumeBwdPlayback){
				if (!mDirection && lastFrameSent <outFrame->timestamp){
					LOG_ERROR <<"Sending newer frame:" << "lastFrameSent: "<<lastFrameSent<<"This frame:"<<outFrame->timestamp;
				}
				else if (mDirection && lastFrameSent >outFrame->timestamp){
					LOG_ERROR <<"Sending older frame:" << "lastFrameSent: "<<lastFrameSent<<"This frame:"<<outFrame->timestamp;
				}
				send(frames);
				auto myId = Module::getId();
				lastFrameSent = outFrame->timestamp;
				if (lastFrameSent == 0){
					LOG_ERROR<<"something is wrong";
				}
			}
		}
		if(playbackSpeed == 2 || playbackSpeed == 4)
		{
			if(!framesToSkip)
			{
				framesToSkip = (currentFps * playbackSpeed) / currentFps ;
			}
			framesToSkip--;
		}
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

bool H264Decoder::handleCommand(Command::CommandType type, frame_sp& frame)
{
	if (type == Command::CommandType::DecoderPlaybackSpeed)
	{
		DecoderPlaybackSpeed cmd;
		getCommand(cmd, frame);
		currentFps = cmd.playbackFps;
		playbackSpeed = cmd.playbackSpeed;
		gop = cmd.gop;
		
		if(playbackSpeed == 2 || playbackSpeed == 4)
		{
			if(previousFps >= currentFps * 8)
			{
				flushQue();
			}
			framesToSkip = (currentFps *playbackSpeed) / currentFps - 1;
		}
		else if(playbackSpeed == 8 || playbackSpeed == 16 || playbackSpeed == 32)
		{
			flushQue();
			framesToSkip = 0;
			if((currentFps * playbackSpeed) / gop > currentFps)
			{
				iFramesToSkip = ((currentFps * playbackSpeed) / gop) / currentFps;
			}
		}
		else
		{
			if(previousFps >= currentFps * 8)
			{
				flushQue();
			}
			framesToSkip = 0;
		}
		LOG_INFO << "frames to skip in decoder in decoder = " << framesToSkip << " fps = " << currentFps * playbackSpeed;
		previousFps = currentFps;
	}
	if (type == Command::CommandType::Seek)
	{
		incomingFramesTSQSize = incomingFramesTSQ.size();
		resumePlayback = false;
		LOG_INFO<<"Pausing decoder";
	}
	else if (type == Command::CommandType::Relay)
	{
		Module::handleCommand(type, frame);
	}
	return true;
}

bool H264Decoder::shouldTriggerSOS()
{
	return mShouldTriggerSOS;
}

bool H264Decoder::processEOS(string& pinId)
{
	//THIS HAS BEEN COMMENTED IN NVR - BECAUSE EOS IS SENT FROM MP4READER WHICH COMES TO DECODER AND THE FOLLOWING PROCESS IS NOT REQUIRED IN NVR. 
	
	// auto frame = frame_sp(new EmptyFrame());
	// mDetail->compute(frame->data(), frame->size(), frame->timestamp);
	// LOG_ERROR << "processes sos " ;
	//mShouldTriggerSOS = true;
	return true;
}

void H264Decoder::flushQue()
{
	Module::flushQue();
}
