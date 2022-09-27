#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <stdafx.h>
#include <map>
#include "Frame.h"
#include "MultimediaQueue.h"
#include "Logger.h"
#include "H264Utils.h"
#include "EncodedImageMetadata.h"
#include "H264Metadata.h"

class QueueClass
{
public:
	typedef std::map<uint64_t, frame_container> MultimediaQueueMap;
	MultimediaQueueMap mQueue;
	virtual bool enqueue(frame_container& frames, uint32_t lowerWaterMark, uint32_t upperWaterMark, bool isMapDelayInTime, bool  pushNext) { return true; }
	const_buffer spsBuffer;
	const_buffer ppsBuffer;
};

class JpegQueueClass : public QueueClass
{

public:
	~JpegQueueClass()
	{}

	bool enqueue(frame_container& frames, uint32_t lowerWaterMark, uint32_t upperWaterMark, bool isMapDelayInTime, bool  pushNext)
	{	//	Here the frame_containers are inserted into the map
		uint64_t largestTimeStamp = 0;
		for (auto it = frames.cbegin(); it != frames.cend(); it++)
		{
			mQueue.insert({ it->second->timestamp, frames });
			if (largestTimeStamp < it->second->timestamp)
			{
				largestTimeStamp = it->second->timestamp;
			}
		}
		if (isMapDelayInTime) // If the lower and upper watermark are given in time
		{
			if ((largestTimeStamp - mQueue.begin()->first > lowerWaterMark) && (pushNext == true))
			{
				mQueue.erase(mQueue.begin()->first);
			}

			else if ((largestTimeStamp - mQueue.begin()->first > upperWaterMark) && (pushNext == false))
			{
				auto it = mQueue.begin();
				auto lastElement = mQueue.end();
				lastElement--;
				auto lastElementTimeStamp = lastElement->first;
				while (it != mQueue.end())
				{
					if ((lastElementTimeStamp - it->first) < lowerWaterMark)
					{
						break;
					}
					auto itr = it;
					++it;
					mQueue.erase(itr->first);
				}
				pushNext = true;
			};
		}
		else // If the lower and upper water mark are given in number of frames
		{
			if ((mQueue.size() > lowerWaterMark) && (pushNext == true))
			{
				mQueue.erase(mQueue.begin()->first);
			}

			else if ((mQueue.size() > upperWaterMark) && (pushNext == false))
			{
				auto it = mQueue.begin();
				while (it != mQueue.end())
				{
					if (mQueue.size() < lowerWaterMark)
					{
						break;
					}
					auto itr = it;
					++it;
					mQueue.erase(itr->first);
				}
				pushNext = true;
			};
		}
		return true;
	}

};

class H264QueueClass : public QueueClass
{

public:
	~H264QueueClass()
	{}

	bool enqueue(frame_container& frames, uint32_t lowerWaterMark, uint32_t upperWaterMark, bool isMapDelayInTime, bool  pushNext)
	{	//	Here the frame_containers are inserted into the map
		uint64_t largestTimeStamp = 0;
		for (auto it = frames.cbegin(); it != frames.cend(); it++)
		{
			auto frame = it->second;
			mutable_buffer& h264Frame = *(frame.get());
			auto ret = H264Utils::parseNalu(h264Frame);
			tie(typeFound, inFrame, spsBuff, ppsBuff) = ret;
			if (spsBuff.size() != 0)
			{
				m_headerFrame = frame;
				spsBuffer = spsBuff;
				ppsBuffer = ppsBuff;
			}
			mQueue.insert({ it->second->timestamp, frames });
			if (largestTimeStamp < it->second->timestamp)
			{
				largestTimeStamp = it->second->timestamp;
			}
		}
		BOOST_LOG_TRIVIAL(info) << "queue size = " << mQueue.size();
		if (isMapDelayInTime) // If the lower and upper watermark are given in time
		{
			if ((largestTimeStamp - mQueue.begin()->first > lowerWaterMark) && (pushNext == true))
			{
				mQueue.erase(mQueue.begin()->first);
				auto it = mQueue.begin();
				while (it != mQueue.end())
				{
					auto itr = it;
					++it;
					auto frame = itr->second.begin()->second;
					mutable_buffer& h264Frame = *(frame.get());
					auto ret = H264Utils::parseNalu(h264Frame);			
					tie(typeFound, inFrame, spsBuff, ppsBuff) = ret;
					
					if (typeFound == H264Utils::H264_NAL_TYPE::H264_NAL_TYPE_IDR_SLICE)
					{
						break;
					}
					mQueue.erase(itr->first);
				}

			}

			else if ((largestTimeStamp - mQueue.begin()->first > upperWaterMark) && (pushNext == false))
			{
				auto lastElement = mQueue.end();
				lastElement--;
				auto lastElementTimeStamp = lastElement->first;
				mQueue.erase(mQueue.begin()->first);
				auto it = mQueue.begin();
				while (it != mQueue.end())
				{

					auto itr = it;
					++it;
					auto frame = itr->second.begin()->second;
					mutable_buffer& h264Frame = *(frame.get());
					auto ret = H264Utils::parseNalu(h264Frame);
					tie(typeFound, inFrame, spsBuff, ppsBuff) = ret;
					
					if (typeFound == H264Utils::H264_NAL_TYPE::H264_NAL_TYPE_SEQ_PARAM)
					{
						break;
					}
					mQueue.erase(itr->first);

					if ((lastElementTimeStamp - it->first) < lowerWaterMark)
					{
						break;
					}

				}
				pushNext = true;
			};
		}
		else // If the lower and upper water mark are given in number of frames
		{
			if ((mQueue.size() > lowerWaterMark) && (pushNext == true))
			{
				mQueue.erase(mQueue.begin()->first);
			}

			if ((mQueue.size() > upperWaterMark) && (pushNext == false))
			{
				auto it = mQueue.begin();
				while (it != mQueue.end())
				{
					if (mQueue.size() < lowerWaterMark)
					{
						break;
					}
					auto itr = it;
					++it;
					auto frame = it->second.begin()->second;
					mutable_buffer& h264Frame = *(frame.get());
					auto ret = H264Utils::parseNalu(h264Frame);
					tie(typeFound, inFrame, spsBuff, ppsBuff) = ret;
					if (typeFound == H264Utils::H264_NAL_TYPE::H264_NAL_TYPE_IDR_SLICE)
					{
						break;
					}
					mQueue.erase(itr->first);
				}
				pushNext = true;
			};
		}
		return true;
	}
protected:
	frame_sp m_headerFrame;
	const_buffer spsBuff;
	const_buffer ppsBuff;
	const_buffer inFrame;
	short typeFound;
};

//State Design begins here
class Export : public State {
public:
	Export(State::StateType type) : State(StateType::EXPORT) {}

	bool handleExport(uint64_t& queryStart, uint64_t& queryEnd, bool& timeReset, mQueueMap& queueMap, uint64_t& endTimeSaved,std::string mOutputPinId) override { return true; }
};

class ExportJpeg : public Export
{
public:
	ExportJpeg() : Export(StateType::EXPORT) {}
	ExportJpeg(boost::shared_ptr<QueueClass> queueObj, std::function<bool(frame_container& frames, bool forceBlockingPush)> _send) : Export(StateType::EXPORT) {
		send = _send;
		queueObject = queueObj;
	}

	bool exportSend(frame_container& frames)
	{
		send(frames, false);
		return true;
	}

	bool handleExport(uint64_t& queryStart, uint64_t& queryEnd, bool& timeReset, mQueueMap& queueMap, uint64_t& endTimeSaved,std::string mOutputPinId) override
	{
		auto tOld = queueMap.begin()->first;
		auto temp = queueMap.end();
		temp--;
		auto tNew = temp->first;
		queryEnd = queryEnd - 1;
		if ((queryStart < tOld) && (queueMap.upper_bound(queryEnd) != queueMap.end()))
		{
			queryStart = tOld;
			timeReset = true;
			return true;
		}
		else if ((queryEnd > tNew) && (queueMap.upper_bound(queryStart) != queueMap.end()))
		{
			if (tNew >= queryEnd)
			{
				timeReset = true;
			}
			queryEnd = tNew;

			return true;
		}
		else
		{
			timeReset = true;
			return true;
		}
	}
};

class ExportH264 : public Export
{
public:
	ExportH264() : Export(StateType::EXPORT) {}
	ExportH264(boost::shared_ptr<QueueClass> queueObj, std::function<bool(frame_container& frames, bool forceBlockingPush)> _send, std::function<frame_sp(size_t size,string pinID)> _makeFrame, std::function<std::string(int type)> _getInputPinIdByType) : Export(StateType::EXPORT) {
		getInputPinIdByType = _getInputPinIdByType;
		makeFrame = _makeFrame;
		send = _send;
		queueObject = queueObj;
	}

	bool exportSend(frame_container& frames)
	{
		if (count == 0)
		{
			//mOutputPinId = "MultimediaQueue_2_pin_1";

			auto tempFrame= makeFrame(frames.begin()->second->size() + queueObject->spsBuffer.size() + queueObject->ppsBuffer.size() + 8, mOutputPinId);
			//tempFrame.get()
			boost::asio::mutable_buffer tempBuffer(tempFrame->data(), tempFrame->size());
			prependSpsPps(tempBuffer,frames);
			frame_container IFrameToSend;
			IFrameToSend.insert(make_pair(frames.begin()->first, tempFrame));
			IFrameToSend.begin()->second->timestamp = frames.begin()->second->timestamp;
			IFrameToSend.begin()->second->fIndex2 = frames.begin()->second->fIndex2;
			send(IFrameToSend, false);
			count++;
		}
		else
		{
			send(frames, false);
		}
		//send(frames, false);
		return true;
	}

	void prependSpsPps(boost::asio::mutable_buffer& iFrameBuffer, frame_container& iFrame)
	{
		frame_sp &iFrameData =  iFrame.begin()->second;
		boost::asio::mutable_buffer tBuffer(iFrameData->data(), iFrameData->size());

		char NaluSeprator[4] = { 00 ,00, 00 ,01 };
		auto nalu = reinterpret_cast<uint8_t*>(NaluSeprator);
		memcpy(iFrameBuffer.data(), nalu, 4);
		iFrameBuffer += 4;
		memcpy(iFrameBuffer.data(), queueObject->spsBuffer.data(), queueObject->spsBuffer.size());
		iFrameBuffer += queueObject->spsBuffer.size();
		memcpy(iFrameBuffer.data(), nalu, 4);
		iFrameBuffer += 4;
		memcpy(iFrameBuffer.data(), queueObject->ppsBuffer.data(), queueObject->ppsBuffer.size());
		iFrameBuffer += queueObject->ppsBuffer.size();
		memcpy(iFrameBuffer.data(), tBuffer.data(), tBuffer.size());
		
	}

	bool handleExport(uint64_t& queryStart, uint64_t& queryEnd, bool& timeReset, mQueueMap& queueMap, uint64_t& endTimeSaved, std::string _mOutputPinId) override
	{
		mOutputPinId = _mOutputPinId;
		auto tOld = queueMap.begin()->first;
		auto temp = queueMap.end();
		temp--;
		auto tNew = temp->first;
		queryEnd = queryEnd - 1;
		bool foundIFrame = false;
		bool updateQueryStart = true;
		if ((queryStart < tOld) && (queueMap.upper_bound(queryEnd) != queueMap.end())) //queryStart is past and queryEnd is present in map
		{
			queryStart = tOld;
			//frame_sp tmpFrame;
			/*auto frame = queueMap.begin()->second.begin()->second;
			boost::asio::mutable_buffer tmpBuffer(frame->data(), frame->size());
			prependSpsPps(tmpBuffer);*/
			updateAllTimestamp = false;
			if (isBFrameEnabled) // If B frame is present we must export GOP's till I-Frame
			{
				for (auto it = queueMap.begin(); it != queueMap.end(); it++)
				{
					auto frame = it->second.begin()->second;
					mutable_buffer& h264Frame = *(frame.get());
					auto ret = H264Utils::parseNalu(h264Frame);
					tie(typeFound, inFrame, spsBuff, ppsBuff) = ret;
					if ((typeFound == H264Utils::H264_NAL_TYPE::H264_NAL_TYPE_IDR_SLICE) && (it->first > queryEnd))
					{
						queryEnd = it->first;
						endTimeSaved = queryEnd;
						timeReset = true;
						foundIFrame = true;
						break;
					}
				}	
				if (!foundIFrame) // If I-Frame is not present then send till p frame.
				{
					queryEnd = tNew;
					endTimeSaved = tNew;
				}
				return true;
			}
			else
			{
				timeReset = true;
				return true;
			}
		}
		else if ((queryEnd > tNew) && (queueMap.upper_bound(queryStart) != queueMap.end())) //queryStart is present in map and queryEnd is in future
		{
			if (tNew >= queryEnd)
			{
				timeReset = true;
			}

			for (auto it = queueMap.lower_bound(queryStart); it != queueMap.begin(); it--)
			{
				auto frame = it->second.begin()->second;
				mutable_buffer& h264Frame = *(frame.get());
				auto ret = H264Utils::parseNalu(h264Frame);
				tie(typeFound, inFrame, spsBuff, ppsBuff) = ret;
				if (typeFound == H264Utils::H264_NAL_TYPE::H264_NAL_TYPE_IDR_SLICE)
				{
					queryStart = it->first;
					queryStart--;
					break;
				}
			}
			queryEnd = tNew;
			updateQueryStart = false;
			return true;
		}
		else //Both queryStart and queryEnd are present in the queue
		{
			if (updateQueryStart)
			{
				if (!isProcessCall)
				{
					for (auto it = queueMap.lower_bound(queryStart); it != queueMap.begin(); it--) // Setting queryStart time
					{
						auto frame = it->second.begin()->second;
						mutable_buffer& h264Frame = *(frame.get());
						auto ret = H264Utils::parseNalu(h264Frame);
						tie(typeFound, inFrame, spsBuff, ppsBuff) = ret;
						if (typeFound == H264Utils::H264_NAL_TYPE::H264_NAL_TYPE_IDR_SLICE)
						{
							queryStart = it->first; 
							queryStart--;
							break;
						}
					}
				}
			}

			auto tOld = queueMap.begin()->first;
			auto temp = queueMap.end();
			temp--;
			auto tNew = temp->first;
			// Setting queryEnd time
			if ((isBFrameEnabled) && (updateAllTimestamp))
			{
				auto tempEndTime = endTimeSaved;
				for (auto it = queueMap.begin(); it != queueMap.end(); it++)
				{
					auto frame = it->second.begin()->second;
					mutable_buffer& h264Frame = *(frame.get());
					auto ret = H264Utils::parseNalu(h264Frame);
					tie(typeFound, inFrame, spsBuff, ppsBuff) = ret;
					if ((typeFound == H264Utils::H264_NAL_TYPE::H264_NAL_TYPE_IDR_SLICE) && (it->first > tempEndTime))
					{
						if ((it->first > endTimeSaved) && (updateEndTimeSaved))
						{
							if (isProcessCall)
							{
								queryStart = endTimeSaved;
							}
							endTimeSaved = it->first;
							timeReset = true;
							updateEndTimeSaved = false;
							break;
						}
						break;
					}
				}
				return true;
			}
			timeReset = true;
			return true;
		}
	}
private:
	std::function<frame_sp(size_t size, string pinID)> makeFrame;
	uint64_t IFrameTS = 0;
	bool updateAllTimestamp = true;
	bool updateEndTimeSaved = true;
	const_buffer ppsBuffer;
	const_buffer spsBuff;
	const_buffer ppsBuff;
	const_buffer inFrame;
	short typeFound;
	string mOutputPinId;
	int count = 0;
};


class Waiting : public State {
public:
	Waiting() : State(State::StateType::WAITING) {}
	Waiting(boost::shared_ptr<QueueClass> queueObj) : State(State::StateType::WAITING) {
		queueObject = queueObj;
	}
	bool handleExport(uint64_t& queryStart, uint64_t& queryEnd, bool& timeReset, mQueueMap& queueMap, uint64_t& endTimeSaved, std::string mOutputPinId) override
	{
		BOOST_LOG_TRIVIAL(info) << "WAITING STATE : THE FRAMES ARE IN FUTURE!! WE ARE WAITING FOR THEM..";
		return true;
	}
};

class Idle : public State {
public:
	Idle() : State(StateType::IDLE) {}
	Idle(boost::shared_ptr<QueueClass> queueObj) : State(StateType::IDLE) {
		queueObject = queueObj;
	}
	bool handleExport(uint64_t& queryStart, uint64_t& queryEnd, bool& timeReset, mQueueMap& queueMap, uint64_t& endTimeSaved, std::string mOutputPinId) override
	{
		//The code will not come here
		return true;
	}
};

MultimediaQueue::MultimediaQueue(MultimediaQueueProps _props) :Module(TRANSFORM, "MultimediaQueue", _props), mProps(_props)
{
	mState.reset(new State());
}

bool MultimediaQueue::validateInputPins()
{
	return true;
}

bool MultimediaQueue::validateOutputPins()
{
	return true;
}

bool MultimediaQueue::validateInputOutputPins()
{
	return Module::validateInputOutputPins();
}

void MultimediaQueue::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);
	auto mType = metadata->getFrameType();
	framemetadata_sp mOutputMetadata;
	if (mType == FrameMetadata::H264_DATA)
	{
		auto rawMetadata = FrameMetadataFactory::downcast<H264Metadata>(metadata);
		auto height = rawMetadata->getHeight();
		auto width = rawMetadata->getWidth();
		mOutputMetadata = framemetadata_sp(new H264Metadata(width, height));
	}
	else if(mType == FrameMetadata::ENCODED_IMAGE)
	{
		auto rawMetadata = FrameMetadataFactory::downcast<EncodedImageMetadata>(metadata);
		auto height = rawMetadata->getHeight();
		auto width = rawMetadata->getWidth();
		mOutputMetadata = framemetadata_sp(new EncodedImageMetadata(width,height));
	}
	else if (mFrameType == FrameMetadata::RAW_IMAGE)
	{
		auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
		auto height = rawMetadata->getHeight();
		auto width = rawMetadata->getWidth();
		mOutputMetadata = framemetadata_sp(new RawImageMetadata(width,height, rawMetadata->getImageType(), rawMetadata->getType(), 0, rawMetadata->getDepth(), FrameMetadata::HOST, true));
	}
	mOutputMetadata->copyHint(*metadata.get());
	mOutputPinId = pinId;
	addOutputPin(metadata, pinId);
	
}

//bool MultimediaQueue::setNext(boost::shared_ptr<Module> next, bool open, bool sieve)
//{
//	//auto inputpinidmetadatamap = getinputmetadatabytype(framemetadata::h264_data);
//	//addoutputpin(inputpinidmetadatamap);
//	return Module::setNext(next, open, false, sieve);
//}

bool MultimediaQueue::init()
{
	if (!Module::init())
	{
		return false;
	}
	auto inputPinIdMetadataMap = getInputMetadata();

	for (auto const& element : inputPinIdMetadataMap)
	{
		auto& metadata = element.second;
		mFrameType = metadata->getFrameType();
		if ((mFrameType == FrameMetadata::FrameType::ENCODED_IMAGE) || (mFrameType == FrameMetadata::FrameType::RAW_IMAGE))
		{
			mState->queueObject.reset(new JpegQueueClass());
		}

		else if (mFrameType == FrameMetadata::FrameType::H264_DATA)
		{
			mState->queueObject.reset(new H264QueueClass());
		}
	}
	mState.reset(new Idle(mState->queueObject));

	return true;
}

bool MultimediaQueue::term()
{
	mState.reset();
	return Module::term();
}

void MultimediaQueue::getQueueBoundaryTS(uint64_t& tOld, uint64_t& tNew)
{
	auto queueMap = mState->queueObject->mQueue;
	tOld = queueMap.begin()->first;
	auto tempIT = queueMap.end();
	tempIT--;
	tNew = tempIT->first;
}

void MultimediaQueue::setState(uint64_t tStart, uint64_t tEnd)
{
	uint64_t tOld, tNew = 0;
	getQueueBoundaryTS(tOld, tNew);

	//Checking conditions to determine the new state and transitions to it.

	if (tEnd < tOld)
	{
		BOOST_LOG_TRIVIAL(info) << "IDLE STATE : MAYBE THE FRAMES HAVE PASSED THE QUEUE";
		mState.reset(new Idle(mState->queueObject));
	}
	else if (tStart > tNew)
	{
		mState.reset(new Waiting(mState->queueObject));
	}
	else
	{
		if ((mFrameType == FrameMetadata::FrameType::ENCODED_IMAGE) || (mFrameType == FrameMetadata::FrameType::RAW_IMAGE))
		{
			mState.reset(new ExportJpeg(mState->queueObject,
				[&](frame_container& frames, bool forceBlockingPush = false)
			{return send(frames, forceBlockingPush); }));
		}
		else if (mFrameType == FrameMetadata::FrameType::H264_DATA)
		{ 
			mState.reset(new ExportH264(mState->queueObject,
				[&](frame_container& frames, bool forceBlockingPush = false)
			{return send(frames, forceBlockingPush);},
				[&](size_t size, string pinID)
			{ return makeFrame(size,pinID); },
				[&](int type)
			{return getInputPinIdByType(type); }));
		}
	}

}

bool MultimediaQueue::handleCommand(Command::CommandType type, frame_sp& frame)
{
	if (type == Command::CommandType::MultimediaQueue)
	{
		MultimediaQueueCommand cmd;
		getCommand(cmd, frame);
		setState(cmd.startTime, cmd.endTime);
		queryStartTime = cmd.startTime;
		startTimeSaved = cmd.startTime;
		queryEndTime = cmd.endTime;
		endTimeSaved = cmd.endTime;

		bool reset = false;
		pushNext = true;
		if (mState->Type == State::EXPORT)
		{
			mState->handleExport(queryStartTime, queryEndTime, reset, mState->queueObject->mQueue, endTimeSaved, mOutputPinId);
			for (auto it = mState->queueObject->mQueue.begin(); it != mState->queueObject->mQueue.end(); it++)
			{
				if (((it->first) >= queryStartTime) && (((it->first) <= queryEndTime)))
				{
					if (isNextModuleQueFull())
					{
						pushNext = false;
						queryStartTime = it->first;
						queryStartTime--;
						BOOST_LOG_TRIVIAL(info) << "The Queue of Next Module is full, waiting for queue to be free";
						return true;
					}
					else
					{
						mState->exportSend(it->second);
					}
				}
			}
		}

		if (mState->Type == mState->EXPORT)
		{
			uint64_t tOld = 0, tNew = 0;
			getQueueBoundaryTS(tOld, tNew);
			if (endTimeSaved > tNew)
			{
				reset = false;
			}
			queryStartTime = tNew;
			
		}
		if (reset)
		{
			queryStartTime = 0;
			queryEndTime = 0;
			setState(queryStartTime, queryEndTime);
		}
		return true;
	}
}

bool MultimediaQueue::allowFrames(uint64_t& ts, uint64_t& te)
{
	if (mState->Type != mState->EXPORT)
	{
		MultimediaQueueCommand cmd;
		cmd.startTime = ts;
		cmd.endTime = te;
		return queueCommand(cmd);
	};
	return true;
}

bool MultimediaQueue::process(frame_container& frames)
{
	mState->queueObject->enqueue(frames, mProps.lowerWaterMark, mProps.upperWaterMark, mProps.isMapDelayInTime, pushNext);
	if (mState->Type == State::EXPORT)
	{
		uint64_t tOld, tNew = 0;
		getQueueBoundaryTS(tOld, tNew);
		queryEndTime = tNew;
	}

	if (mState->Type == State::WAITING)
	{
		setState(queryStartTime, queryEndTime);
	}

	if (mState->Type == State::EXPORT)
	{
		mState->isProcessCall = true;
		mState->handleExport(queryStartTime, queryEndTime, reset, mState->queueObject->mQueue, endTimeSaved, mOutputPinId);
		for (auto it = mState->queueObject->mQueue.begin(); it != mState->queueObject->mQueue.end(); it++)
		{
			if (((it->first) >= (queryStartTime + 1)) && (((it->first) <= (endTimeSaved))))
			{
				if (isNextModuleQueFull())
				{
					pushNext = false;
					queryStartTime = it->first;
					queryStartTime--;
					BOOST_LOG_TRIVIAL(info) << "The Queue of Next Module is full, waiting for some space to be free";
					return true;
				}
				else
				{
					mState->exportSend(it->second);
				}
			}

		}
	}
	if (mState->Type == State::EXPORT)
	{
		uint64_t tOld, tNew = 0;
		getQueueBoundaryTS(tOld, tNew);
		if (queryEndTime > tNew);
		{
			reset = false;
		}
		queryStartTime = tNew;
	}

	if (reset)
	{
		queryStartTime = 0;
		queryEndTime = 0;
		setState(queryStartTime, queryEndTime);
	}
	return true;
}

bool MultimediaQueue::handlePropsChange(frame_sp& frame)
{
	if (mState->Type != State::EXPORT)
	{
		MultimediaQueueProps props(10, 5, false);
		auto ret = Module::handlePropsChange(frame, props);
		return ret;
	}

	else
	{
		BOOST_LOG_TRIVIAL(info) << "Currently in export state, wait until export is completed";
		return true;
	}
}

MultimediaQueueProps MultimediaQueue::getProps()
{
	fillProps(mProps);
	return mProps;
}

void  MultimediaQueue::setProps(MultimediaQueueProps _props)
{
	if (mState->Type != State::EXPORT)
	{
		mProps = _props;
		Module::addPropsToQueue(mProps);
	}
	else
	{
		BOOST_LOG_TRIVIAL(info) << "Currently in export state, wait until export is completed";
	}
}