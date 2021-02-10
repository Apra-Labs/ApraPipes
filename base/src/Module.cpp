#include "stdafx.h"
#include <boost/asio.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/foreach.hpp>

#include <chrono>
#include <boost/lexical_cast.hpp>
#include "Module.h"
#include "AIPExceptions.h"
#include "Frame.h"
#include "FrameContainerQueue.h"
#include "FrameMetadata.h"

#include "PaceMaker.h"

class Module::Profiler
{
	using sys_clock = std::chrono::system_clock;

public:
	Profiler(string& id, bool _shouldLog, int _printFrequency, std::function<string()> _getPoolHealthRecord): moduleId(id), shouldLog(_shouldLog), mPipelineFps(0), printFrequency(_printFrequency)
	{
		getPoolHealthRecord = _getPoolHealthRecord;
	}

	void setShouldLog(bool _shouldLog)
	{
		shouldLog = _shouldLog;
	}

	virtual ~Profiler()
	{

	}

	void startPipelineLap()
	{

		pipelineStart = sys_clock::now();
		processingStart = pipelineStart;
	}

	void startProcessingLap()
	{

		processingStart = sys_clock::now();
	}

	void endLap(size_t _queSize)
	{
		sys_clock::time_point end = sys_clock::now();
		std::chrono::nanoseconds diff = end - pipelineStart;
		totalPipelineDuration += diff.count();
		diff = end - processingStart;
		totalProcessingDuration += diff.count();
		queSize += _queSize;

		counter += 1;
		if (counter % printFrequency == 0)
		{
			auto processingDurationInSeconds = totalProcessingDuration / 1000000000.0;
			double processingFps = printFrequency / processingDurationInSeconds;
			auto pipelineDurationInSeconds = totalPipelineDuration / 1000000000.0;
			double pipelineFps = printFrequency / pipelineDurationInSeconds;
			auto idleWaitingTime = pipelineDurationInSeconds - processingDurationInSeconds;

			if (shouldLog)
			{
				LOG_INFO << moduleId << " processed<" << printFrequency << "> frames. Pipeline Time<" << pipelineDurationInSeconds << "> PipelineAvgFps<" << std::setprecision(5) << pipelineFps << "> Processing Time<" << processingDurationInSeconds << "> ProcessingAvgFps<" << std::setprecision(5) << processingFps << "> AvgQue<" << std::setprecision(5) << (queSize / printFrequency) << "> IdleTime<" << idleWaitingTime << "> Que<" << _queSize << "> " << getPoolHealthRecord();
			}

			totalPipelineDuration = 0;
			totalProcessingDuration = 0;
			queSize = 0;

			mPipelineFps = pipelineFps;
		}
	}

	uint64_t getTickCounter()
	{
		return counter;
	}

	double getPipelineFps()
	{
		return mPipelineFps;
	}

	void resetStats()
	{
		mPipelineFps = 0;
	}

private:
	string moduleId;
	sys_clock::time_point processingStart;
	sys_clock::time_point pipelineStart;
	int printFrequency;
	uint64_t counter = 0;
	double totalProcessingDuration = 0;
	double totalPipelineDuration = 0;
	double queSize = 0;
	bool shouldLog = false;
	std::function<string()> getPoolHealthRecord;

	double mPipelineFps;
};

Module::Module(Kind nature, string name, ModuleProps _props) :mRunning(false), mPlay(true), mForceStepCount(0), mStopCount(0), mForwardPins(0), myNature(nature), myName(name), mSkipIndex(0) {
	static int moduleCounter = 0;
	moduleCounter += 1;
	myId = name + "_" + std::to_string(moduleCounter);
		
	mpFrameFactory.reset(new FrameFactory(_props.maxConcurrentFrames));
	mpCommandFactory.reset(new FrameFactory());
	
	mQue.reset(new FrameContainerQueue(_props.qlen));
	
	onStepFail=boost::bind(&Module::ignore,this,0);
		
	pacer = boost::shared_ptr<PaceMaker>(new PaceMaker(_props.fps));
	auto tempId = getId();
	mProfiler.reset(new Profiler(tempId, _props.logHealth, _props.logHealthFrequency, [&]() -> std::string {return mpFrameFactory->getPoolHealthRecord(); }));
	if(_props.skipN > _props.skipD)
	{
		throw AIPException(AIP_ROI_OUTOFRANGE, "skipN <= skipD");
	}
	mProps.reset(new ModuleProps(_props)); // saving for restoring later

	mCommandMetadata.reset(new FrameMetadata(FrameMetadata::FrameType::COMMAND));
	mPropsChangeMetadata.reset(new FrameMetadata(FrameMetadata::FrameType::PROPS_CHANGE));	
}
Module::~Module() 
{

}

bool Module::term()
{	
	mQue->clear();
	// in case of cyclic dependency - one module holds the reference of the other and hence they never get freed
	mModules.clear();
	mProfiler->resetStats(); 

	return true;
}

double Module::getPipelineFps()
{
	return mProfiler->getPipelineFps();
}

uint64_t Module::getTickCounter()
{
	return mProfiler->getTickCounter();
}

void Module::setProps(ModuleProps& props)
{
	if (props.qlen != mProps->qlen)
	{
		throw AIPException(AIP_NOTIMPLEMENTED, string("qlen cannot be changed"));
	}
	
	pacer->setFps(props.fps);
	mProfiler->setShouldLog(props.logHealth);	
	if(props.skipN > props.skipD)
	{
		// processing all
		props.skipN = 0;
		props.skipD = 1;
	}
	mProps.reset(new ModuleProps(props));
}

ModuleProps Module::getProps()
{
	return *mProps.get();
}

void Module::fillProps(ModuleProps& props)
{
	props.fps = mProps->fps;
	props.qlen = mProps->qlen;
	props.logHealth = mProps->logHealth;
}

string Module::addOutputPin(framemetadata_sp& metadata)
{
	std::string pinId = myId + "_pin_" + std::to_string(mOutputPinIdMetadataMap.size() + 1);
	addOutputPin(metadata, pinId);	

	return pinId;
}

void Module::addOutputPin(framemetadata_sp& metadata, string& pinId)
{
	if (mOutputPinIdMetadataMap.find(pinId) != mOutputPinIdMetadataMap.end())
	{
		// key alread exist exception
		auto msg = "<" + getId() + "> pinId<" + pinId + "> Already Exist. Please give unique name.";
		throw AIPException(AIP_UNIQUE_CONSTRAINT_FAILED, msg);
	}
	   
	mOutputPinIdMetadataMap[pinId] = metadata;	
	
	if (!validateOutputPins())
	{
		mOutputPinIdMetadataMap.erase(pinId);
		auto msg = "<" + getId() + "> Output Pins Validation Failed.";
		throw AIPException(AIP_PINS_VALIDATION_FAILED, msg);
	}
}

bool Module::setNext(boost::shared_ptr<Module> next, vector<string>& pinIdArr, bool open, bool isFeedback)
{
	if (next->getNature() < this->getNature())
	{
		LOG_ERROR << "Can not connect these modules " << this->getId() << " -> " << next->getId();
		return false;
	}

	if (pinIdArr.size() == 0)
	{
		LOG_ERROR << "No Pins to connect. " << this->getId() << " -> " << next->getId();
		return false;
	}
		
	auto nextModuleId = next->getId();
	if (mModules.find(nextModuleId) != mModules.end())
	{
		LOG_ERROR << "<" << getId() << "> Connection for <" << nextModuleId << " > already done.";
		return false;
	}
	mModules[nextModuleId] = next;
	mConnections.insert(make_pair(nextModuleId, boost::container::deque<string>()));
	
	for (auto& pinId : pinIdArr) 
	{
		if (mOutputPinIdMetadataMap.find(pinId) == mOutputPinIdMetadataMap.end()) {
			auto msg = "pinId<" + pinId + "> doesn't exist in <" + this->getId() + ">";
			mModules.erase(nextModuleId);
			mConnections.erase(nextModuleId);
			throw AIPException(AIP_PIN_NOTFOUND, msg);
		}

		framemetadata_sp metadata = mOutputPinIdMetadataMap[pinId];

		// Set input meta here
		try
		{
			next->addInputPin(metadata, pinId, isFeedback); // addInputPin throws exception from validateInputPins
		}
		catch (AIP_Exception& exception)
		{
			mModules.erase(nextModuleId);
			mConnections.erase(nextModuleId);
			throw exception;
		}
		catch (...)
		{
			mModules.erase(nextModuleId);
			mConnections.erase(nextModuleId);
			LOG_FATAL << "";
			throw AIPException(AIP_FATAL, "<" + getId() + "> addInputPin. PinId<" + pinId + ">. Unknown exception.");
		}

		// add next module here
		mConnections[nextModuleId].push_back(pinId);
	}

	mRelay[nextModuleId] = open;

	return true;
}

bool Module::setNext(boost::shared_ptr<Module> next, bool open)
{
	return setNext(next, open, false);
}

bool Module::setNext(boost::shared_ptr<Module> next, bool open, bool isFeedback)
{
	pair<string, framemetadata_sp> me; // map element
	vector<string> pinIdArr;
	BOOST_FOREACH (me, mOutputPinIdMetadataMap)
	{
		pinIdArr.push_back(me.first);
	}

	// sending all the outputpins
	return setNext(next, pinIdArr, open, isFeedback);
}

bool Module::setNext(boost::shared_ptr<Module> next, vector<string> &pinIdArr, bool open)
{
	return setNext(next, pinIdArr, open, false);
}

bool Module::addFeedback(boost::shared_ptr<Module> next, vector<string> &pinIdArr, bool open)
{
	return setNext(next, pinIdArr, open, true);
}

bool Module::addFeedback(boost::shared_ptr<Module> next, bool open)
{
	return setNext(next, open, true);
}

void Module::addInputPin(framemetadata_sp &metadata, string &pinId, bool isFeedback)
{
	addInputPin(metadata, pinId);
	if (isFeedback)
	{
		mForwardPins--;
		mInputPinsDirection[pinId] = false; // feedback
	}
}

void Module::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	if (mInputPinIdMetadataMap.find(pinId) != mInputPinIdMetadataMap.end())
	{
		auto msg = "<" + getId() + "> pinId <" + pinId + "> already added for <" + getId() + ">";
		throw AIPException(AIP_UNIQUE_CONSTRAINT_FAILED, msg);
	}

	mInputPinIdMetadataMap[pinId] = metadata;

	if (!validateInputPins()) {
		mInputPinIdMetadataMap.erase(pinId);
		auto msg = "Input Pins Validation Failed. <" + getId() + ">";
		throw AIPException(AIP_PINS_VALIDATION_FAILED, msg);
	}

	mForwardPins++;
	mInputPinsDirection[pinId] = true; // forward
}

bool Module::isFeedbackEnabled(std::string& moduleId)
{
	auto& pinIdArr = mConnections[moduleId];
	auto childModule = mModules[moduleId];
	for (auto itr = pinIdArr.begin(); itr != pinIdArr.end(); itr++)
	{
		auto& pinId = *itr;
		if(childModule->mInputPinsDirection[pinId])
		{
			// forward pin found - so feedback not enabled
			return false;
		}
	}

	return true;
}

bool Module::validateInputPins()
{
	if (myNature == SOURCE && getNumberOfInputPins() == 0)
	{		
		return true;
	}

	return false;
}

bool Module::validateOutputPins()
{
	if (myNature == SINK && getNumberOfOutputPins() == 0)
	{
		return true;
	}

	return false;
}

framemetadata_sp Module::getFirstInputMetadata()
{
	return mInputPinIdMetadataMap.begin()->second;
}

framemetadata_sp Module::getFirstOutputMetadata()
{
	return mOutputPinIdMetadataMap.begin()->second;
}

boost::container::deque<boost::shared_ptr<Module>> Module::getConnectedModules()
{
	boost::container::deque<boost::shared_ptr<Module>> nextModules;

	for (map<string, boost::shared_ptr<Module>>::const_iterator it = mModules.cbegin(); it != mModules.cend(); ++it)
	{	
		auto pModule = it->second;
		nextModules.push_back(pModule);		
	}

	return nextModules;
}

bool Module::init()
{
	auto ret = validateInputOutputPins();
	if (!ret)
	{
		return ret;
	}

	mQue->accept();
	if (mModules.size() == 1 && mProps->quePushStrategyType == QuePushStrategy::NON_BLOCKING_ALL_OR_NONE)
	{
		mProps->quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	}
	mQuePushStrategy = QuePushStrategy::getStrategy(mProps->quePushStrategyType, myId);
	// loop all the downstream modules and set the que
	for (map<string, boost::shared_ptr<Module>>::const_iterator it = mModules.begin(); it != mModules.end(); ++it)
	{
		auto pModule = it->second;
		auto que = pModule->getQue();
		mQuePushStrategy->addQue(it->first, que);
	}

	if (myNature == TRANSFORM && getNumberOfInputPins() == 1 && getNumberOfOutputPins() == 1)
	{
		// propagate hint
		// currently propagating if 1 input and 1 output

		auto in = getFirstInputMetadata();
		auto out = getFirstOutputMetadata();
		out->copyHint(*in.get());
	}

	mStopCount = 0;

	mFIndexStrategy = FIndexStrategy::create(mProps->fIndexStrategyType);

	return ret;
}

bool Module::push(frame_container frameContainer) {
	mQue->push(frameContainer);
	return true;
}


bool Module::try_push(frame_container frameContainer)
{	
	auto rc = mQue->try_push(frameContainer);
	return rc;
}

frame_container Module::try_pop()
{
	return mQue->try_pop();
}

frame_container Module::pop()
{
	return mQue->pop();
}

bool Module::isFull() {
	bool ret = false;
	map<string, boost::shared_ptr<Module>> mModules;
	for (auto it = mModules.cbegin(); it != mModules.end(); it++)
	{
		if (it->second->isFull())
		{
			ret = true;
			break;
		}
	}
	
	return ret;
}

bool Module::send(frame_container& frames, bool forceBlockingPush) {
	// mFindex may be propagated for EOS, EOP, Command, PropsChange also - which is wrong
	uint64_t fIndex = 0; 
	uint64_t timestamp = 0;
	if (frames.size() != 0)
	{
		if (myNature == TRANSFORM && getNumberOfInputPins() == 1)
		{
			// propagating fIndex2
			auto pinId = getInputMetadata().begin()->first;
			if (frames.find(pinId) != frames.end())
			{
				auto fIndex2 = frames[pinId]->fIndex2;
				for (auto me = mOutputPinIdMetadataMap.cbegin(); me != mOutputPinIdMetadataMap.cend(); me++)
				{
					if (frames.find(me->first) != frames.end())
					{
						frames[me->first]->fIndex2 = fIndex2;
					}
				}
			}
		}

		if (myNature != SOURCE)
		{		
			// first input pin
			auto pinId = getInputMetadata().begin()->first;
			if (frames.find(pinId) != frames.end())
			{
				fIndex = frames[pinId]->fIndex;
				timestamp = frames[pinId]->timestamp;
			}
			else
			{
				// try output pins - muxer comes here
				for (auto me = mOutputPinIdMetadataMap.cbegin(); me != mOutputPinIdMetadataMap.cend(); me++)
				{
					auto& pinId = me->first;
					if (frames.find(pinId) != frames.end())
					{
						fIndex = frames[pinId]->fIndex;
						timestamp = frames[pinId]->timestamp;
						break;
					}
				}
			}							
		}
		else
		{
			// try for all output pins
			for(auto me = mOutputPinIdMetadataMap.cbegin(); me != mOutputPinIdMetadataMap.cend(); me++) 
			{
				auto& pinId = me->first;
				if (frames.find(pinId) != frames.end())
				{
					fIndex = frames[pinId]->fIndex;
					timestamp = frames[pinId]->timestamp;
					break;
				}				
			}
		}
	}

	fIndex = mFIndexStrategy->getFIndex(fIndex);

	for (auto it = frames.cbegin(); it != frames.cend(); it++)
	{
		// change only for output pins - add condition here
		it->second->fIndex = fIndex;
		it->second->timestamp = timestamp;
	}

	auto ret = true;
	// loop over all the modules and send
	for (Connections::const_iterator it = mConnections.begin(); it != mConnections.end(); it++)
	{
		auto& nextModuleId = it->first;
		if (!mRelay[nextModuleId] && !forceBlockingPush)
		{
			// This is dangerous - the callers may assume that all the frames go through - but since it is relay - they wont go through
			// so using forceBlockingPush to open the relay for important messages
			// currently only EOS and EOP frames can break the relay
			continue;
		}

		auto pinsArr = it->second;
		frame_container requiredPins;
		
		for (auto i = pinsArr.begin(); i != pinsArr.end(); i++)
		{
			auto pinId = *i;
			if (frames.find(pinId) == frames.end())
			{
				// pinId not found
				continue;
			}
			requiredPins.insert(make_pair(pinId, frames[pinId])); // only required pins map is created
		}

		if (requiredPins.size() == 0)
		{
			// no pins found
			continue;
		}

		// next module push		
		if (!forceBlockingPush)
		{
			mQuePushStrategy->push(nextModuleId, requiredPins);
		}
		else
		{
			mModules[nextModuleId]->push(requiredPins);
		}
	}

	return mQuePushStrategy->flush();
}

boost_deque<frame_sp> Module::getFrames(frame_container& frames)
{
	boost_deque<frame_sp> frames_arr;
	for (frame_container::const_iterator it = frames.begin(); it != frames.end(); it++)
	{
		frames_arr.push_back(it->second);
	}

	return frames_arr;
}

string getPinIdByType(int type, metadata_by_pin& metadataMap)
{
	pair<string, framemetadata_sp> me; // map element		
	BOOST_FOREACH(me, metadataMap) {
		if (me.second->getFrameType() == type)
		{
			return me.first;
		}
	}

	return "";
}

vector<string> Module::getAllOutputPinsByType(int type)
{
	vector<string> pins;

	pair<string, framemetadata_sp> me; // map element		
	BOOST_FOREACH(me, mOutputPinIdMetadataMap) {
		if (me.second->getFrameType() == type)
		{
			pins.push_back(me.first);
		}
	}

	return pins;
}

string Module::getInputPinIdByType(int type)
{
	return getPinIdByType(type, mOutputPinIdMetadataMap);
}

string Module::getOutputPinIdByType(int type)
{
	return getPinIdByType(type, mOutputPinIdMetadataMap);
}

framemetadata_sp getMetadataByType(int type, metadata_by_pin& metadataMap)
{
	pair<string, framemetadata_sp> me; // map element		
	BOOST_FOREACH(me, metadataMap) {
		if (me.second->getFrameType() == type)
		{
			return me.second;
		}
	}

	return framemetadata_sp();
}

int getNumberOfPinsByType(int type, metadata_by_pin& metadataMap)
{
	int count = 0;
	pair<string, framemetadata_sp> me; // map element		
	BOOST_FOREACH(me, metadataMap) {
		if (me.second->getFrameType() == type)
		{
			count += 1;
		}
	}

	return count;
}

framemetadata_sp Module::getInputMetadataByType(int type)
{
	return getMetadataByType(type, mInputPinIdMetadataMap);
}

framemetadata_sp Module::getOutputMetadataByType(int type)
{
	return getMetadataByType(type, mOutputPinIdMetadataMap);
}

int Module::getNumberOfInputsByType(int type)
{
	return getNumberOfPinsByType(type, mInputPinIdMetadataMap);
}

int Module::getNumberOfOutputsByType(int type)
{
	return getNumberOfPinsByType(type, mOutputPinIdMetadataMap);
}

bool Module::isMetadataEmpty(framemetadata_sp& metatata)
{
	return !metatata.get();
}

bool Module::isFrameEmpty(frame_sp& frame)
{
	return !frame.get();
}

frame_sp Module::getFrameByType(frame_container& frames, int frameType)
{
	// This returns only the first matched frametype 
	// remmeber the map is ordered by pin ids
	for (auto it = frames.cbegin(); it != frames.cend(); it++)
	{
		auto frame = it->second;
		if (frame->getMetadata()->getFrameType() == frameType)
		{
			return frame;
		}
	}

	return frame_sp();
}

frame_sp Module::makeFrame(size_t size)
{
	return makeFrame(size, mOutputPinIdMetadataMap.begin()->second);
}

frame_sp Module::makeFrame(size_t size, string& pinId) {	
	return makeFrame(size, mOutputPinIdMetadataMap[pinId]);
}

frame_sp Module::makeCommandFrame(size_t size, framemetadata_sp& metadata) {
	auto frame = mpCommandFactory->create(size, mpCommandFactory, metadata->getMemType());
	frame->setMetadata(metadata);
	
	return frame;
}

frame_sp Module::makeFrame(size_t size, framemetadata_sp& metadata) {
	auto frame = mpFrameFactory->create(size, mpFrameFactory, metadata->getMemType());
	if (frame.get())
	{
		frame->setMetadata(metadata);
	}
	return frame;
}

frame_sp Module::makeFrame(buffer_sp& buffer, size_t& size, framemetadata_sp& metadata)
{	
	auto frame = mpFrameFactory->create(buffer, size, mpFrameFactory, metadata->getMemType());	
	frame->setMetadata(metadata);
	return frame;
}

buffer_sp Module::makeBuffer(size_t size, FrameMetadata::MemType memType)
{
	return mpFrameFactory->createBuffer(size, mpFrameFactory, memType);	
}

frame_sp Module::getEOSFrame()
{
	return mpFrameFactory->getEOSFrame();
}

frame_sp Module::getEmptyFrame()
{
	return mpFrameFactory->getEmptyFrame();
}

void Module::operator()() { 
	run(); 
}
bool Module::run()
{
	LOG_INFO << "Starting " << myId << " on " << myThread.get_id() ;
	mRunning = true;
	handlePausePlay(mPlay);
	while (mRunning)
	{
		if (!step())
		{
			stop_onStepfail();
			break;
		}
	}
	LOG_INFO << "Ending " << myId << " on " << myThread.get_id() ;
	term(); //my job is done
	return true;
}

bool isMetadatset(metadata_by_pin& metadataMap)
{
	bool bSet = true;

	pair<string, framemetadata_sp> me; // map element		
	BOOST_FOREACH(me, metadataMap) {
		if (!me.second->isSet()) {
			bSet = false;
			break;
		}
	}

	return bSet;
}

bool Module::shouldTriggerSOS()
{			
	if (!isMetadatset(mInputPinIdMetadataMap) || !isMetadatset(mOutputPinIdMetadataMap))
	{
		return true;
	}	

	return false;
}

bool Module::play(bool play)
{
	if (!mRunning)
	{
		// comes here if module is not running in a thread
		// comes here when pipeline is started with run_all_threaded_withpause
		return handlePausePlay(play);
	}

	auto metadata = framemetadata_sp(new PausePlayMetadata());
	auto frame = makeFrame(metadata->getDataSize(), metadata);

	auto buffer = (unsigned char*)frame->data();
	memset(frame->data(), play, 1);
	
	// add to que
	frame_container frames;
	frames.insert(make_pair("pause_play", frame));
	if (!Module::try_push(frames))
	{
		LOG_ERROR << "failed to push play command to the que";
	}

	return true;
}

bool Module::queueStep()
{
	auto cmd = StepCommand();
	return queueCommand(cmd);
}

bool Module::relay(boost::shared_ptr<Module> next, bool open)
{
	auto nextModuleId = next->getId();
	if (mModules.find(nextModuleId) == mModules.end())
	{
		LOG_ERROR << "<" << getId() << "> Connection for <" << nextModuleId << " > doesn't exist.";
		return false;
	}

	auto cmd = RelayCommand(nextModuleId, open);
	return queueCommand(cmd);
}

bool Module::processSourceQue()
{
	frame_container frames;
	while ((frames = mQue->try_pop()).size())	
	{		
		auto it = frames.cbegin();
		while (it != frames.cend())
		{
			auto frame = it->second;
			auto pinId = it->first;
			it++;

			if (frame->isPausePlay())
			{
				auto buffer = (unsigned char*)frame->data();
				auto play = buffer[0] ? true : false;
				handlePausePlay(play);
			}
			else if (frame->isPropsChange())
			{
				handlePropsChange(frame);
			}
			else if (frame->isCommand())
			{
				auto cmdType = NoneCommand::getCommandType(frame->data(), frame->size());
				handleCommand(cmdType, frame);
			}
			else if (frame->isEoP())
			{
				handleStop();
				return false;
			}
			else
			{
				LOG_ERROR << frame->getMetadata()->getFrameType() << "<> not handled";
			}
		}
	}

	return true;
}

bool Module::handlePausePlay(bool play)
{
	mPlay = play;
	notifyPlay(mPlay);

	return true;
}

bool Module::step()
{		
	bool ret = false;	
	if (myNature == SOURCE)
	{
		if(!processSourceQue())
		{
			return true;
		}
		bool forceStep = shouldForceStep();

		pacer->start();
		
		if (mPlay || forceStep)
		{
			mProfiler->startPipelineLap();
			ret = produce();
			mProfiler->endLap(0);
		}
		else
		{
			ret = true;
			// ret false will kill the thread
		}
		
		pacer->end();
	}
	else
	{
		mProfiler->startPipelineLap();
		auto frames = mQue->pop();	
		preProcessNonSource(frames);
		
		if (frames.size() == 0 || shouldSkip())
		{
			// it can come here only if frames.erase from processEOS or processSOS or processEoP or isPropsChange() or isCommand()
			return true;
		}

		mProfiler->startProcessingLap();
		ret = stepNonSource(frames);		
		mProfiler->endLap(mQue->size());
	}
		
	return ret;
}

void Module::sendEOS()
{	
	if (myNature == SINK)
	{
		return;
	}

	frame_container frames;
	auto frame = frame_sp(new EoSFrame());
	pair<string, framemetadata_sp> me; // map element	
	BOOST_FOREACH(me, mOutputPinIdMetadataMap) {
		frames.insert(make_pair(me.first, frame));
	}

	send(frames, true);
}

bool Module::preProcessNonSource(frame_container& frames)
{
	auto bTriggerSOS = shouldTriggerSOS(); // donot calculate every time - store the state when condition changes

	bool eosEncountered = false;
	auto it = frames.cbegin();
	while (it != frames.cend())
	{
		// increase the iterator manually

		auto frame = it->second;
		auto pinId = it->first;
		it++;
		if (frame->isEOS())
		{
			// EOS Strategy
			// should we send all frames at a shot or 1 by 1 ?
			processEOS(pinId);
			if (!eosEncountered)
			{
				sendEOS(); // propagating  eosframe with every eos encountered
			}
			frames.erase(pinId);
			eosEncountered = true;
			continue;
		}

		if (frame->isPropsChange())
		{
			if (!handlePropsChange(frame))
			{
				throw AIPException(AIP_FATAL, string("Handle PropsChange failed"));
			}
			frames.erase(pinId);
			continue;
		}

		if (frame->isEoP())
		{
			handleStop();
			frames.erase(pinId);
			continue;
		}

		if (frame->isCommand())
		{
			auto cmdType = NoneCommand::getCommandType(frame->data(), frame->size());
			handleCommand(cmdType, frame);
			frames.erase(pinId);
			continue;
		}

		if (!bTriggerSOS)
		{
			// framemetadata is set. No action required			
			continue;
		}

		// new framemetadata_sp can be created - example JPEGDecoderNVJPEG
		mInputPinIdMetadataMap[pinId] = frame->getMetadata();
		if (!processSOS(frame))
		{
			// remove frame from frames because it is still not ready to process frames		
			frames.erase(pinId);
		}
		// bug: outputmetadata can also be updated ? give a set function
	}

	return true;
}

bool Module::stepNonSource(frame_container& frames)
{	
	bool ret = true;
	try
	{
		ret = process(frames);
	}
	catch (AIP_Exception&)
	{
		// assuming already logged
	}
	catch (...)
	{
		LOG_FATAL << getId() << "<> Unknown exception. Catching throw";
	}

	return ret;
}

bool Module::addEoPFrame(frame_container& frames)
{
	pair<string, framemetadata_sp> me; // map element	
	BOOST_FOREACH(me, mOutputPinIdMetadataMap) {
		auto frame = frame_sp(new EoPFrame());
		frame->setMetadata(me.second);
		frames.insert(make_pair(me.first, frame));
	}
}

bool Module::handleStop()
{
	// force stop is required
	if (mRunning == false)
	{
		return true;
	}
	mStopCount++;
	if (myNature != SOURCE && mStopCount != mForwardPins)
	{
		return true;
	}
	if (myNature == SINK)
	{
		mRunning = false;
		return true;
	}

	sendEoPFrame();

	mRunning = false;

	return true;
}

void Module::sendEoPFrame()
{
	frame_container frames;
	addEoPFrame(frames);

	send(frames, true);
}

bool Module::stop()
{
	frame_container frames;
	addEoPFrame(frames);

	Module::push(frames);

	return true;
}

void Module::adaptQueue(boost::shared_ptr<FrameContainerQueueAdapter> queAdapter)
{
	queAdapter->adapt(mQue);
	mQue = queAdapter;
}

void Module::ignore(int times) {
	static int observed = 0;
	observed++;
	if (observed >= times && times>0)
	{
		LOG_TRACE << "stopping due to step failure ";
		observed = 0;
		handleStop();
	}
	
} 

void Module::stop_onStepfail() {
	LOG_ERROR << "Stopping " << myId << " due to step failure ";
	handleStop();
}

void Module::emit_event(unsigned short eventID)
{
	if (!event_consumer.empty())
	{
		event_consumer(this, eventID);
		event_consumer.clear(); // we can only fire once. 
	}
}

void Module::emit_fatal(unsigned short eventID)
{
	if (!fatal_event_consumer.empty())
	{
		//we have a handler... let's trigger it
		fatal_event_consumer(this, eventID);
	}
	else {
		//we dont have a handler let's kill this thread
		std::string msg("Fatal error in module ");
		LOG_FATAL << "FATAL error in module : " << myName;
		msg += myName;
		msg += " Event ID "; 
		msg += std::to_string(eventID);
		throw AIPException(AIP_FATAL, msg);
	}
}

void Module::register_consumer(boost::function<void(Module*, unsigned short)> consumer, bool bFatal /*= false*/)
{
	(bFatal) ? (fatal_event_consumer = consumer) : (event_consumer = consumer);
}

bool Module::handlePropsChange(frame_sp& frame)
{
	throw AIPException(AIP_NOTIMPLEMENTED, "Props Change for not implemented");
}

bool Module::handleCommand(Command::CommandType type, frame_sp& frame)
{
	switch (type)
	{
	case Command::Relay:
	{
		RelayCommand cmd;
		getCommand(cmd, frame);

		mRelay[cmd.nextModuleId] = cmd.open;
	}
		break;
	case Command::Step:
	{
		// call step
		mForceStepCount++;
	}
		break;
	default:
		throw AIPException(AIP_NOTIMPLEMENTED, "Command Handler for <" + to_string(type) + "> not implemented");
	}
	
	return true;
}

bool Module::shouldForceStep()
{
	auto forceStep = mForceStepCount > 0;
	if (forceStep)
	{
		mForceStepCount--;
	}

	return forceStep;
}

bool Module::shouldSkip()
{	
	if (mProps->skipN == 0)
	{
		return false;
	}
	
	if (mProps->skipN == mProps->skipD)
	{
		return true;
	}

	auto skip = true;

	if (mSkipIndex <= 0 || mSkipIndex > mProps->skipD)
	{
		mSkipIndex = mProps->skipD;
	}

	if (mSkipIndex > mProps->skipN)
	{
		skip = false;
	}

	mSkipIndex--;

	return skip;
}