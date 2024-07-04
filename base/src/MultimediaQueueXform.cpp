#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <stdafx.h>
#include <map>
#include <thread>
#include "Frame.h"
#include "MultimediaQueueXform.h"
#include "Logger.h"
#include "H264Utils.h"
#include "EncodedImageMetadata.h"
#include "H264Metadata.h"
#include "FrameContainerQueue.h"
#include "AbsControlModule.h"

class FramesQueue
{
public:
	typedef std::map<uint64_t, frame_container> MultimediaQueueXformMap;
	MultimediaQueueXformMap mQueue;
	virtual bool enqueue(frame_container& frames, bool  pushToNextModule) { return true; }
	const_buffer spsBuffer;
	const_buffer ppsBuffer;
	uint32_t lowerWaterMark = 0;
	uint32_t upperWaterMark = 0;
	bool isMapDelayInTime = true;
};

class IndependentFramesQueue : public FramesQueue
{

public:
	IndependentFramesQueue(uint32_t _lowerWaterMark, uint32_t _upperWaterMark, bool _isMapDelayInTime)
	{
		lowerWaterMark = _lowerWaterMark;
		upperWaterMark = _upperWaterMark;
		isMapDelayInTime = _isMapDelayInTime;
	}
	~IndependentFramesQueue()
	{}

	bool enqueue(frame_container& frames, bool  pushToNextModule)
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
			if ((largestTimeStamp - mQueue.begin()->first > lowerWaterMark) && (pushToNextModule))
			{
				mQueue.erase(mQueue.begin()->first);
			}

			else if ((largestTimeStamp - mQueue.begin()->first > upperWaterMark) && (pushToNextModule == false))
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
				pushToNextModule = true;
			};
		}

		else // If the lower and upper water mark are given in number of frames
		{
			if ((mQueue.size() > lowerWaterMark) && (pushToNextModule == true))
			{
				mQueue.erase(mQueue.begin()->first);
			}

			else if ((mQueue.size() > upperWaterMark) && (pushToNextModule == false))
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
				pushToNextModule = true;
			};
		}
		return true;
	}

};

class GroupedFramesQueue : public FramesQueue
{

public:
	GroupedFramesQueue(uint32_t _lowerWaterMark, uint32_t _upperWaterMark, bool _isMapDelayInTime)
	{
		lowerWaterMark = _lowerWaterMark;
		upperWaterMark = _upperWaterMark;
		isMapDelayInTime = _isMapDelayInTime;
	}
	~GroupedFramesQueue()
	{}

	bool enqueue(frame_container& frames, bool  pushToNextModule)
	{	//	Here the frame_containers are inserted into the map
		uint64_t largestTimeStamp = 0;
		for (auto it = frames.cbegin(); it != frames.cend(); it++)
		{
			auto frame = it->second;
			auto mFrameBuffer = const_buffer(frame->data(), frame->size());
			auto ret = H264Utils::parseNalu(mFrameBuffer);
			tie(typeFound, spsBuff, ppsBuff) = ret;

			//BOOST_LOG_TRIVIAL(info) << "I-FRAME" << typeFound;

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
			if ((largestTimeStamp - mQueue.begin()->first > lowerWaterMark) && (pushToNextModule == true))
			{
				mQueue.erase(mQueue.begin()->first);
				auto it = mQueue.begin();
				while (it != mQueue.end())
				{
					auto itr = it;
					++it;
					auto frame = itr->second.begin()->second;
					auto mFrameBuffer = const_buffer(frame->data(), frame->size());
					auto ret = H264Utils::parseNalu(mFrameBuffer);
					tie(typeFound, spsBuff, ppsBuff) = ret;

					if (typeFound == H264Utils::H264_NAL_TYPE::H264_NAL_TYPE_IDR_SLICE)
					{
						break;
					}
					mQueue.erase(itr->first);
				}

			}

			else if ((largestTimeStamp - mQueue.begin()->first > upperWaterMark) && (pushToNextModule == false))
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
					auto mFrameBuffer = const_buffer(frame->data(), frame->size());
					auto ret = H264Utils::parseNalu(mFrameBuffer);
					tie(typeFound, spsBuff, ppsBuff) = ret;

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
				pushToNextModule = true;
			};
		}

		else // If the lower and upper water mark are given in number of frames
		{
			if ((mQueue.size() > lowerWaterMark) && (pushToNextModule == true))
			{
				mQueue.erase(mQueue.begin()->first);
			}

			if ((mQueue.size() > upperWaterMark) && (pushToNextModule == false))
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
					auto mFrameBuffer = const_buffer(frame->data(), frame->size());
					auto ret = H264Utils::parseNalu(mFrameBuffer);
					tie(typeFound, spsBuff, ppsBuff) = ret;

					if (typeFound == H264Utils::H264_NAL_TYPE::H264_NAL_TYPE_IDR_SLICE)
					{
						break;
					}
					mQueue.erase(itr->first);
				}
				pushToNextModule = true;
			};
		}
		return true;
	}
protected:
	frame_sp m_headerFrame;
	const_buffer spsBuff;
	const_buffer ppsBuff;
	short typeFound;
};

class State {
public:
	boost::shared_ptr<FramesQueue> queueObject;
	State() {}
	virtual ~State() {}
	typedef std::map<uint64_t, frame_container> mQueueMap;
	virtual bool handleExport(uint64_t& queryStart, uint64_t& queryEnd, bool& timeReset, mQueueMap& mQueue, uint64_t& endTimeSaved) { return true; };
	virtual bool exportSend(frame_container& frames) { return true; };
	std::function<bool(frame_container& frames, bool forceBlockingPush)> send;
	std::function<std::string(int type)> getInputPinIdByType;

	bool isBFrameEnabled = true;
	bool isProcessCall = false;
	enum StateType
	{
		IDLE = 0,
		WAITING,
		EXPORT
	};

	State(StateType type_)
	{
		Type = type_;
	}
	StateType Type = StateType::IDLE;
};

//State Design begins here
class ExportQState : public State {
public:
	ExportQState(State::StateType type) : State(StateType::EXPORT) {}

	bool handleExport(uint64_t& queryStart, uint64_t& queryEnd, bool& timeReset, mQueueMap& queueMap, uint64_t& endTimeSaved) override { return true; }
};

class ExportJpeg : public ExportQState
{
public:
	ExportJpeg() : ExportQState(StateType::EXPORT) {}
	ExportJpeg(boost::shared_ptr<FramesQueue> queueObj, std::function<bool(frame_container& frames, bool forceBlockingPush)> _send) : ExportQState(StateType::EXPORT) {
		send = _send;
		queueObject = queueObj;
	}

	bool exportSend(frame_container& frames)
	{
		send(frames, false);
		return true;
	}

	bool handleExport(uint64_t& queryStart, uint64_t& queryEnd, bool& timeReset, mQueueMap& queueMap, uint64_t& endTimeSaved) override
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

class ExportH264 : public ExportQState
{
public:
	ExportH264() : ExportQState(StateType::EXPORT) {}
	ExportH264(boost::shared_ptr<FramesQueue> queueObj, std::function<bool(frame_container& frames, bool forceBlockingPush)> _send, std::function<frame_sp(size_t size, string pinID)> _makeFrame, std::function<std::string(int type)> _getInputPinIdByType, std::string _mOutputPinId) : ExportQState(StateType::EXPORT) {
		getInputPinIdByType = _getInputPinIdByType;
		makeFrame = _makeFrame;
		send = _send;
		queueObject = queueObj;
		mOutputPinId = _mOutputPinId;
	}

	bool exportSend(frame_container& frames)
	{
		send(frames, false);
		return true;
	}

	bool handleExport(uint64_t& queryStart, uint64_t& queryEnd, bool& timeReset, mQueueMap& queueMap, uint64_t& endTimeSaved) override
	{
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

			if (isBFrameEnabled) // If B frame is present we must export GOP's till I-Frame
			{
				for (auto it = queueMap.begin(); it != queueMap.end(); it++)
				{
					auto frame = it->second.begin()->second;
					auto mFrameBuffer = const_buffer(frame->data(), frame->size());
					auto ret = H264Utils::parseNalu(mFrameBuffer);
					tie(typeFound, spsBuff, ppsBuff) = ret;
					if ((typeFound == H264Utils::H264_NAL_TYPE::H264_NAL_TYPE_IDR_SLICE) && (it->first > queryEnd))
					{
						queryEnd = it->first;
						endTimeSaved = queryEnd;
						timeReset = true;
						foundIFrame = true;
						updateAllTimestamp = false;
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

			for (auto it = queueMap.lower_bound(queryStart);; it--)
			{
				auto frame = it->second.begin()->second;
				auto mFrameBuffer = const_buffer(frame->data(), frame->size());
				auto ret = H264Utils::parseNalu(mFrameBuffer);
				tie(typeFound, spsBuff, ppsBuff) = ret;

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
					for (auto it = queueMap.lower_bound(queryStart);; it--) // Setting queryStart time
					{
						auto frame = it->second.begin()->second;
						auto mFrameBuffer = const_buffer(frame->data(), frame->size());
						auto ret = H264Utils::parseNalu(mFrameBuffer);
						tie(typeFound, spsBuff, ppsBuff) = ret;

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
					auto mFrameBuffer = const_buffer(frame->data(), frame->size());
					auto ret = H264Utils::parseNalu(mFrameBuffer);
					tie(typeFound, spsBuff, ppsBuff) = ret;

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
	short typeFound;
	string mOutputPinId;
	int count = 0;
};


class Waiting : public State {
public:
	Waiting() : State(State::StateType::WAITING) {}
	Waiting(boost::shared_ptr<FramesQueue> queueObj) : State(State::StateType::WAITING) {
		queueObject = queueObj;
	}
	bool handleExport(uint64_t& queryStart, uint64_t& queryEnd, bool& timeReset, mQueueMap& queueMap, uint64_t& endTimeSaved) override
	{
		BOOST_LOG_TRIVIAL(info) << "WAITING STATE : THE FRAMES ARE IN FUTURE!! WE ARE WAITING FOR THEM..";
		return true;
	}
};

class Idle : public State {
public:
	Idle() : State(StateType::IDLE) {}
	Idle(boost::shared_ptr<FramesQueue> queueObj) : State(StateType::IDLE) {
		queueObject = queueObj;
	}
	bool handleExport(uint64_t& queryStart, uint64_t& queryEnd, bool& timeReset, mQueueMap& queueMap, uint64_t& endTimeSaved) override
	{
		//The code will not come here
		return true;
	}
};

MultimediaQueueXform::MultimediaQueueXform(MultimediaQueueXformProps _props) :Module(TRANSFORM, "MultimediaQueueXform", _props), mProps(_props)
{
	mState.reset(new State());
}

bool MultimediaQueueXform::validateInputPins()
{
	return true;
}

bool MultimediaQueueXform::validateOutputPins()
{
	return true;
}

bool MultimediaQueueXform::validateInputOutputPins()
{
	return Module::validateInputOutputPins();
}

void MultimediaQueueXform::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);
	mOutputPinId = pinId;
	//addOutputPin(metadata, pinId);
}

bool MultimediaQueueXform::init()
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
		if ((mFrameType == FrameMetadata::FrameType::ENCODED_IMAGE) || (mFrameType == FrameMetadata::FrameType::RAW_IMAGE) || (mFrameType == FrameMetadata::FrameType::RAW_IMAGE_PLANAR))
		{
			mState->queueObject.reset(new IndependentFramesQueue(mProps.lowerWaterMark, mProps.upperWaterMark, mProps.isMapDelayInTime));
		}

		else if (mFrameType == FrameMetadata::FrameType::H264_DATA)
		{
			mState->queueObject.reset(new GroupedFramesQueue(mProps.lowerWaterMark, mProps.upperWaterMark, mProps.isMapDelayInTime));
		}
	}
	mState.reset(new Idle(mState->queueObject));
	myTargetFrameLen = std::chrono::nanoseconds(1000000000 / mProps.mmqFps);
	initialFps = mProps.mmqFps;
	return true;
}

bool MultimediaQueueXform::term()
{
	mState.reset();
	return Module::term();
}

void MultimediaQueueXform::getQueueBoundaryTS(uint64_t& tOld, uint64_t& tNew)
{
	auto queueMap = mState->queueObject->mQueue;
	tOld = queueMap.begin()->first;
	auto tempIT = queueMap.end();
	tempIT--;
	tNew = tempIT->first;
}

void MultimediaQueueXform::setState(uint64_t tStart, uint64_t tEnd)
{
	uint64_t tOld, tNew = 0;
	getQueueBoundaryTS(tOld, tNew);

	//Checking conditions to determine the new state and transitions to it.

	if (tEnd < tOld)
	{
		BOOST_LOG_TRIVIAL(info) << "IDLE STATE : MAYBE THE FRAMES HAVE PASSED THE QUEUE";
		mState.reset(new Idle(mState->queueObject));
		reset = false;
	}

	else if (tStart > tNew)
	{
		mState.reset(new Waiting(mState->queueObject));
	}

	else
	{
		if ((mFrameType == FrameMetadata::FrameType::ENCODED_IMAGE) || (mFrameType == FrameMetadata::FrameType::RAW_IMAGE) || (mFrameType == FrameMetadata::FrameType::RAW_IMAGE_PLANAR))
		{
			mState.reset(new ExportJpeg(mState->queueObject,
				[&](frame_container& frames, bool forceBlockingPush = false)
			{return send(frames, forceBlockingPush); }));
		}

		else if (mFrameType == FrameMetadata::FrameType::H264_DATA)
		{
			mState.reset(new ExportH264(mState->queueObject,
				[&](frame_container& frames, bool forceBlockingPush = false)
			{return send(frames, forceBlockingPush); },
				[&](size_t size, string pinID)
			{ return makeFrame(size, pinID); },
				[&](int type)
			{return getInputPinIdByType(type); }, mOutputPinId));
		}
	}
}

void MultimediaQueueXform::extractFramesAndEnqueue(boost::shared_ptr<FrameContainerQueue>& frameQueue)
{
	//loop over frame container
	if (frameQueue->size())
	{
		frame_container framesContainer;
		auto frames = frameQueue->pop();
		for (auto itr = frames.begin(); itr != frames.end(); itr++)
		{
			if (itr->second->isCommand())
			{
				auto cmdType = NoneCommand::getCommandType(itr->second->data(), itr->second->size());
				handleCommand(cmdType, itr->second);
			}
			else if(itr->second->isPropsChange())
			{
				handlePropsChange(itr->second);
			}
			else
			{
				framesContainer.insert(make_pair(itr->first, itr->second));
			}
		}
		if (!framesContainer.empty())
		{
			mState->queueObject->enqueue(framesContainer, pushToNextModule);
		}
	}
}

boost::shared_ptr<FrameContainerQueue> MultimediaQueueXform::getQue()
{
	return Module::getQue();
}

bool MultimediaQueueXform::handleCommand(Command::CommandType type, frame_sp& frame)
{
	if(type ==  Command::CommandType::DecoderPlaybackSpeed)
	{
		DecoderPlaybackSpeed dCmd;
		getCommand(dCmd, frame);
		setPlaybackSpeed(dCmd.playbackSpeed);
		//setMmqFps(dCmd.playbackFps);
	}
	int fps = mProps.mmqFps * speed;
	LOG_ERROR << "mmq fps is = " << fps;
	myTargetFrameLen = std::chrono::nanoseconds(1000000000 / fps);
	initDone = false;
	if (type == Command::CommandType::ExportMMQ)
	{
        ExportMMQ cmd;
		getCommand(cmd, frame);
		setState(cmd.startTime, cmd.endTime);
		queryStartTime = cmd.startTime;
		startTimeSaved = cmd.startTime;
		queryEndTime = cmd.endTime;
		endTimeSaved = cmd.endTime;
		direction = cmd.direction;
		LOG_INFO << "start time = " << cmd.startTime;
		LOG_INFO << "end time = " <<  cmd.endTime;
		LOG_INFO << "direction = " << cmd.direction;
		LOG_INFO << "state = " << mState->Type;
		LOG_INFO << "mmq begin ts = " << mState->queueObject->mQueue.begin()->first;
		auto itttr =   mState->queueObject->mQueue.end();
		itttr--;
		LOG_INFO << "mmq end ts = " << itttr->first;
		bool reset = false;
		pushToNextModule = true;

		if (mState->Type == State::EXPORT)
		{
			LOG_INFO << "inside state export block";
			mState->handleExport(queryStartTime, queryEndTime, reset, mState->queueObject->mQueue, endTimeSaved);
			State::mQueueMap::iterator it;
			if (direction)
			{
				it = mState->queueObject->mQueue.begin();
			}
			else
			{
				it = mState->queueObject->mQueue.end();
				if (!mState->queueObject->mQueue.empty()){
					it--;
					// it--;
				}
				else{
					LOG_INFO<<"Queue is empty";
				}
			}
			State::mQueueMap::iterator it_last;
			it_last = mState->queueObject->mQueue.end();

			State::mQueueMap::iterator it_first;
			it_first = mState->queueObject->mQueue.begin();
			if (!mState->queueObject->mQueue.empty()){
				it_last--;
				if (direction && (queryStartTime >= it->first) && ( queryStartTime <= it_last->first))
				{
					exportFrames = true;
				}
				else if (!direction && (queryEndTime <= it->first) && ( queryEndTime >= it_first->first))
				{
					exportFrames = true;
				}
			}

			while (!mState->queueObject->mQueue.empty() && exportFrames == true)//&& it != mState->queueObject->mQueue.end()
			{
				if (((it->first) >= queryStartTime) && (((it->first) <= queryEndTime)))
				{
					if (isNextModuleQueFull())
					{
						pushToNextModule = false;
						queryStartTime = it->first;
						queryStartTime--;
						LOG_INFO << "The Queue of Next Module is full, waiting for queue to be free";
						return true;
					}
					else
					{
						if (!initDone)
						{
							myNextWait = myTargetFrameLen;
							frame_begin = sys_clock::now();
							initDone = true;
						}
						frame_container outFrames;
						auto outputId = Module::getOutputPinIdByType(FrameMetadata::RAW_IMAGE_PLANAR);
						outFrames.insert(make_pair(outputId, it->second.begin()->second));
						if (!framesToSkip)
						{
							mState->exportSend(outFrames);
						}
						if(direction && !mState->queueObject->mQueue.empty())
						{
							auto lastItr = mState->queueObject->mQueue.end();
							lastItr--;
							if(lastItr->second.begin()->second->timestamp == it->second.begin()->second->timestamp)
							{
								if(controlModule != nullptr)
								{
									bool goLive = true;
									bool priority = true;
									boost::shared_ptr<AbsControlModule>ctl = boost::dynamic_pointer_cast<AbsControlModule>(controlModule);
									ctl->handleGoLive(goLive, priority);
								}
								exportFrames = false;
								break;
							}
						}
						if (speed != 1 && speed != 0.5)
						{
							if (!framesToSkip)
							{
								framesToSkip = speed;
							}
							framesToSkip--;
						}
						latestFrameExportedFromHandleCmd = it->first;
						std::chrono::nanoseconds frame_len = sys_clock::now() - frame_begin;
						if (myNextWait > frame_len)
						{
							std::this_thread::sleep_for(myNextWait - frame_len);
						}
						myNextWait += myTargetFrameLen;
					}
					if (!((!direction && it == mState->queueObject->mQueue.begin()) || (direction && it == mState->queueObject->mQueue.end())))
					{
						LOG_INFO << "enque frames";
						auto moduleQueue = getQue();
						extractFramesAndEnqueue(moduleQueue);
						it = mState->queueObject->mQueue.find(latestFrameExportedFromHandleCmd);
					}
				}
				if (direction)
				{
					if (mState->queueObject->mQueue.empty())
					{
						break;
					}
					if(it == mState->queueObject->mQueue.end())
					{
						break;
					}
					else
					{
						auto lastItr = mState->queueObject->mQueue.end();
						lastItr--;
						queryEndTime = lastItr->first;
						it++;
					}
				}
				else
				{
					if (it != mState->queueObject->mQueue.end() && it != mState->queueObject->mQueue.begin())
					{
						if(it-- == mState->queueObject->mQueue.begin())
						{
							break;
						}
					}
					if (it == mState->queueObject->mQueue.begin())// || it == mState->queueObject->mQueue.end()
					{
						if (mState->Type != State::IDLE)
						{
							if(controlModule != nullptr)
							{
								// Stubbing the eventual application's control module & the handleExportMMQ method. Might need to implement a custom command. See below. 
								boost::shared_ptr<AbsControlModule> ctl = boost::dynamic_pointer_cast<AbsControlModule>(controlModule);
								ctl->handleMMQExportView(latestFrameExportedFromProcess, 0, direction, true, true);
							}
							exportFrames = false;
						}
						mState->Type = State::IDLE;
						break;
					}
				}
			}
		}
		if (mState->Type == mState->EXPORT)
		{
			uint64_t tOld = 0, tNew = 0;
			tNew = latestFrameExportedFromHandleCmd;

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
		LOG_INFO << "export frames done";
	}
	LOG_INFO <<"RELAY COMMAND WAS HERE";
	return Module::handleCommand(type, frame);
}

bool MultimediaQueueXform::allowFrames(uint64_t& ts, uint64_t& te)
{
	if (mState->Type != mState->EXPORT)
	{
		ExportMMQ cmd;
		cmd.startTime = ts;
		cmd.endTime = te;
		return queueCommand(cmd);
	};
	return true;
}

bool MultimediaQueueXform::process(frame_container& frames)
{
	mState->queueObject->enqueue(frames, pushToNextModule);
	LOG_INFO << frames.begin()->second->timestamp;
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
		mState->handleExport(queryStartTime, queryEndTime, reset, mState->queueObject->mQueue, endTimeSaved);
		State::mQueueMap::iterator it;
		if (direction)
		{
			it = mState->queueObject->mQueue.begin();
		}
		else
		{
			it = mState->queueObject->mQueue.end();
			if (!mState->queueObject->mQueue.empty()){
				it--;
			}
			else
			{
				LOG_ERROR << "Queue is empty";
			}
		}
		State::mQueueMap::iterator it_last;
		it_last = mState->queueObject->mQueue.end();
		if (!mState->queueObject->mQueue.empty()){
			it_last--;
			if (direction && (queryStartTime >= it->first) && ( queryStartTime <= it_last->first))
			{
				exportFrames = true;
			}
			else if (!direction && (queryEndTime >= it->first) && ( queryEndTime <= it_last->first))
			{
				exportFrames = true;
			}
		}
		while (!mState->queueObject->mQueue.empty() && exportFrames == true) //&& it != mState->queueObject->mQueue.end()
		{
			if (((it->first) >= (queryStartTime + 1)) && (((it->first) <= (endTimeSaved))))
			{
				if (isNextModuleQueFull())
				{
					pushToNextModule = false;
					queryStartTime = it->first;
					queryStartTime--;
					LOG_INFO << "The Queue of Next Module is full, waiting for some space to be free";
					return true;
				}
				else
				{
					if (!initDone)
					{
						myNextWait = myTargetFrameLen;
						frame_begin = sys_clock::now();
						initDone = true;
					}

					frame_container outFrames;
					auto outputId = Module::getOutputPinIdByType(FrameMetadata::RAW_IMAGE_PLANAR);

					outFrames.insert(make_pair(outputId, it->second.begin()->second));
					//LOG_ERROR<<"sENDING FROM PROCESS AT TIME "<< it->first;
					if (!framesToSkip)
					{
						// mState->exportSend(outFrames);
					}
					if (speed != 1 && speed != 0.5)
					{
						if (!framesToSkip)
						{
							framesToSkip = (mProps.mmqFps * speed) / mProps.mmqFps;
						}
						framesToSkip--;
					}
					latestFrameExportedFromProcess = it->first;
					std::chrono::nanoseconds frame_len = sys_clock::now() - frame_begin;
					if (myNextWait > frame_len)
					{
						LOG_INFO << "is it sleeping in process";
						std::this_thread::sleep_for(myNextWait - frame_len);
					}
					myNextWait += myTargetFrameLen;
				}
				if (!((!direction && it == mState->queueObject->mQueue.begin()) || (direction && it == mState->queueObject->mQueue.end())))
				{
					auto moduleQueue = getQue();
					extractFramesAndEnqueue(moduleQueue);
					it = mState->queueObject->mQueue.find(latestFrameExportedFromHandleCmd);
				}
			}
			if (direction)
			{
				if (mState->queueObject->mQueue.empty())
				{
					break;
				}
				if (it == mState->queueObject->mQueue.end())
				{
					break;
				}
				else
				{
					it++;
				}
			}
			else
			{
				if (it != mState->queueObject->mQueue.end() && it != mState->queueObject->mQueue.begin())
				{
					if (it-- == mState->queueObject->mQueue.begin())
					{
						break;
					}
				}
				if (it == mState->queueObject->mQueue.begin())
				{
					if (mState->Type != State::IDLE)
					{
						if(controlModule != nullptr)
						{
							// Stubbing the eventual application's control module & the handleExportMMQ method. Might need to implement a custom command. See below. 
							boost::shared_ptr<AbsControlModule> ctl = boost::dynamic_pointer_cast<AbsControlModule>(controlModule);
							ctl->handleMMQExportView(latestFrameExportedFromProcess, 0, direction, true, true);
						}
						exportFrames = false;
					}
					mState->Type = State::IDLE;
					LOG_INFO << "first frame of process = " << latestFrameExportedFromProcess;
					break;
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
	// This part is done only when Control module is connected
	if (controlModule != nullptr)
	{
		// Send commmand to NVRControl module
		if (mState->queueObject->mQueue.size() != 0)
		{
			bool priority = false;
			uint64_t firstTimeStamp;
			auto front = mState->queueObject->mQueue.begin();
			if (front != mState->queueObject->mQueue.end())
			{
				firstTimeStamp = front->first;
			}
			auto back = mState->queueObject->mQueue.crbegin();
			uint64_t lastTimeStamp = back->first;
			// Stubbing the eventual application's control module & the handleExportMMQ method. Might need to implement a custom command. See below. 
			boost::shared_ptr<AbsControlModule>ctl = boost::dynamic_pointer_cast<AbsControlModule>(controlModule);
			ctl->handleSendMMQTSCmd(firstTimeStamp, lastTimeStamp, priority);
		}
		return true;
	}
	return true;
}

void MultimediaQueueXform::setMmqFps(int fps)
{
	mProps.mmqFps = fps;
	mProps.mmqFps--;
}

void MultimediaQueueXform::setPlaybackSpeed(float playbackSpeed)
{
	framesToSkip = 0;
	if(speed != playbackSpeed)
	{
		speed = playbackSpeed;
		int fps = mProps.mmqFps * speed;
		myTargetFrameLen = std::chrono::nanoseconds(1000000000 / fps);
		initDone = false;
		
		if(speed != 1 && speed != 0.5)
		{
			framesToSkip = (mProps.mmqFps * speed) / mProps.mmqFps - 1;
		}
		else
		{
			framesToSkip = 0;
		}
	}
	LOG_INFO << "frames to skip = " << framesToSkip << "speed is  = " << speed;
}

bool MultimediaQueueXform::handlePropsChange(frame_sp& frame)
{
	MultimediaQueueXformProps props(10, 5,2, false);
	auto ret = Module::handlePropsChange(frame, props);

	if (mState->Type != State::EXPORT)
	{
		MultimediaQueueXformProps props(10, 5, false);
		auto ret = Module::handlePropsChange(frame, props);
		return ret;
	}

	else
	{
		BOOST_LOG_TRIVIAL(info) << "Currently in export state, wait until export is completed";
		return true;
	}
}

MultimediaQueueXformProps MultimediaQueueXform::getProps()
{
	fillProps(mProps);
	return mProps;
}

void  MultimediaQueueXform::setProps(MultimediaQueueXformProps _props)
{
	//if (mState->Type != State::EXPORT)
	//{
		mProps = _props;
		Module::addPropsToQueue(mProps, true);
	//}

	//else
	//{
		BOOST_LOG_TRIVIAL(info) << "Currently in export state, wait until export is completed";
	//}
}

void MultimediaQueueXform::stopExportFrames(){
	exportFrames = false;
	}