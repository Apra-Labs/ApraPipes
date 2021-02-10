#pragma once
#include <boost/shared_ptr.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/stream.hpp>
#include "Frame.h"
#include <boost/function.hpp>
#include "BoundBuffer.h"
#include "FrameFactory.h"
#include "CommonDefs.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "QuePushStrategy.h"
#include "FIndexStrategy.h"
#include "Command.h"

using namespace std;

class FrameContainerQueue;
class FrameContainerQueueAdapter;
class PaceMaker;

class ModuleProps
{
public:	
	ModuleProps()
	{
		fps = 60;
		qlen = 20;
		logHealth = false;		
		logHealthFrequency = 1000;
		quePushStrategyType = QuePushStrategy::BLOCKING;
		maxConcurrentFrames = 0;
		fIndexStrategyType = FIndexStrategy::FIndexStrategyType::AUTO_INCREMENT;
	}

	ModuleProps(int _fps)
	{
		fps = _fps;
		qlen = 20;
		logHealth = false;
		logHealthFrequency = 1000;
		quePushStrategyType = QuePushStrategy::BLOCKING;
		maxConcurrentFrames = 0;
		fIndexStrategyType = FIndexStrategy::FIndexStrategyType::AUTO_INCREMENT;
	}

	ModuleProps(int _fps, size_t _qlen, bool _logHealth)
	{
		fps = _fps;
		qlen = _qlen;
		logHealth = _logHealth;
		logHealthFrequency = 1000;
		quePushStrategyType = QuePushStrategy::BLOCKING;
		maxConcurrentFrames = 0;
		fIndexStrategyType = FIndexStrategy::FIndexStrategyType::AUTO_INCREMENT;
	}

	size_t getQLen()
	{
		return qlen;
	}

	virtual size_t getSerializeSize()
	{
		// 1024 is for boost serialize
		return 1024 + sizeof(fps) + sizeof(qlen) + sizeof(logHealth) + sizeof(logHealthFrequency) + sizeof(maxConcurrentFrames) + sizeof(skipN) + sizeof(skipD) + sizeof(quePushStrategyType) + sizeof(fIndexStrategyType);
	}

	int fps; // can be updated during runtime with setProps
	size_t qlen; // run time changing doesn't effect this
	bool logHealth; // can be updated during runtime with setProps
	int logHealthFrequency; // 1000 by default - logs the health stats frequency

	// used for VimbaSource where we want to create the max frames and keep recycling it
	// for the VimbaDrive we announce frames after init - 100/200 
	// see VimbaSource.cpp on how it is used
	size_t maxConcurrentFrames; 

	// 0/1 - skipN == 0 -  don't skip any - process all
	// 1/1 - skipN == skipD - skip all - don't process any
	// 1/2 skips every alternate frame
	// 1/3 skips 1 out of every 3 frames
	// 2/3 skips 2 out of every 3 frames
	// 5/6 skips 5 out of every 6 frames
	// skipD >= skipN
	int skipN = 0; 
	int skipD = 1; 

	QuePushStrategy::QuePushStrategyType quePushStrategyType;
	FIndexStrategy::FIndexStrategyType fIndexStrategyType;
		
private:
	friend class Module;	

	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int /* file_version */) {
		ar & fps & qlen & logHealth & logHealthFrequency & maxConcurrentFrames & skipN & skipD & quePushStrategyType & fIndexStrategyType;
	}
};

class Module {
	
public:
	enum Kind {
		SOURCE,
		TRANSFORM,
		SINK
	};
	enum ModuleState {
		Initialized,
		Running,
		EndOfStreamNormal,
		EndOfStreamSocketError
	};
	Module(Kind nature, string name, ModuleProps _props);
	virtual ~Module();
	Kind getNature() { return myNature; }
	string getName() { return myName; }
	string getId() { return myId; }
	double getPipelineFps();
	uint64_t getTickCounter();
		
	string addOutputPin(framemetadata_sp& metadata); // throw exception
	vector<string> getAllOutputPinsByType(int type);
	void addOutputPin(framemetadata_sp& metadata, string& pinId);
	bool setNext(boost::shared_ptr<Module> next, vector<string>& pinIdArr, bool open = true); 
	bool setNext(boost::shared_ptr<Module> next, bool open = true); // take all the output pins			
	bool addFeedback(boost::shared_ptr<Module> next, vector<string>& pinIdArr, bool open = true); 
	bool addFeedback(boost::shared_ptr<Module> next, bool open = true); // take all the output pins			
	boost_deque<boost::shared_ptr<Module>> getConnectedModules();

	bool relay(boost::shared_ptr<Module> next, bool open);
		
	virtual bool init();
	void operator()(); //to support boost::thread
	virtual bool run();	
	bool play(bool play);
	bool queueStep();
	virtual bool step();
	virtual bool stop();
	virtual bool term();
	virtual bool isFull();

	void adaptQueue(boost::shared_ptr<FrameContainerQueueAdapter> queAdapter);
	
	void register_consumer(boost::function<void(Module*, unsigned short)>, bool bFatal=false);
	boost::shared_ptr<PaceMaker> getPacer() { return pacer; }	
	static frame_sp getFrameByType(frame_container& frames, int frameType);	
protected:
	virtual boost_deque<frame_sp> getFrames(frame_container& frames);	
	virtual bool process(frame_container& frames) { return false; }
	virtual bool processEOS(string& pinId) { return true; } // EOS is propagated in stepNonSource for every encountered EOSFrame - pinId is first stream in the map
	virtual bool processSOS(frame_sp& frame) { return true; } // SOS is Start of Stream	
	virtual bool shouldTriggerSOS();
	virtual bool produce() { return false; }
	bool stepNonSource(frame_container& frames);
	bool preProcessNonSource(frame_container& frames);
	bool isRunning() { return mRunning; }

	ModuleProps getProps();
	void setProps(ModuleProps& props);
	void fillProps(ModuleProps& props);
	template<class T>
	void addPropsToQueue(T& props)
	{
		auto size = props.getSerializeSize();
		auto frame = makeCommandFrame(size, mPropsChangeMetadata);

		// serialize
		serialize<T>(props, frame);
		// add to que
		frame_container frames;
		frames.insert(make_pair("props_change", frame));
		Module::push(frames);
	}
	virtual bool handlePropsChange(frame_sp& frame);
	virtual bool handleCommand(Command::CommandType type, frame_sp& frame);
	template<class T>
	bool handlePropsChange(frame_sp& frame, T& props)
	{
		//deserialize
		deSerialize<T>(props, frame);

		// set props		
		Module::setProps(props);

		return true;
	}

	template<class T>
	bool queueCommand(T& cmd)
	{
		auto size = cmd.getSerializeSize();
		auto frame = makeCommandFrame(size, mCommandMetadata);

		Utils::serialize(cmd, frame->data(), size);

		// add to que
		frame_container frames;
		frames.insert(make_pair("command", frame));
		Module::push(frames);

		return true;
	}

	template<class T>
	void getCommand(T& cmd, frame_sp& frame)
	{
		Utils::deSerialize(cmd, frame->data(), frame->size());
	}
	
	frame_sp makeCommandFrame(size_t size, framemetadata_sp& metadata);
	frame_sp makeFrame(size_t size, framemetadata_sp& metadata);
	frame_sp makeFrame(size_t size, string& pinId);
	frame_sp makeFrame(size_t size); // use only if 1 output pin is there
	frame_sp makeFrame(buffer_sp& buffer, size_t& newSize, framemetadata_sp& metadata);
	buffer_sp makeBuffer(size_t size, FrameMetadata::MemType memType);
	frame_sp getEOSFrame();
	frame_sp getEmptyFrame();
		
	virtual bool send(frame_container& frames, bool forceBlockingPush=false);
	virtual void sendEOS();	
	virtual void sendEoPFrame();
	
	boost::function<void () > onStepFail;
	//various behaviours for stepFail:
	void ignore(int times); //do nothing
	void stop_onStepfail();
	void emit_event(unsigned short eventID); //regular events
	void emit_fatal(unsigned short eventID); //fatal events need a permanent handler
		
	friend class PipeLine;	

	boost::function<void(Module*, unsigned short)> event_consumer;
	boost::function<void(Module*, unsigned short)> fatal_event_consumer;
		
	enum ModuleState module_state;
	void setModuleState(enum ModuleState es) { module_state = es; }
	ModuleState getModuleState() {
		return module_state;
	}

	virtual bool validateInputPins();  // invoked with setInputPin
	virtual bool validateOutputPins(); // invoked with addOutputPin
	virtual bool validateInputOutputPins() { return validateInputPins() && validateOutputPins(); } // invoked during Module::init before anything else
				
	size_t getNumberOfOutputPins() { return mOutputPinIdMetadataMap.size(); }
	size_t getNumberOfInputPins() { return mInputPinIdMetadataMap.size(); }
	framemetadata_sp getFirstInputMetadata();
	framemetadata_sp getFirstOutputMetadata();
	metadata_by_pin& getInputMetadata() { return mInputPinIdMetadataMap; }
	metadata_by_pin& getOutputMetadata() { return mOutputPinIdMetadataMap; }
	framemetadata_sp getInputMetadataByType(int type);
	int getNumberOfInputsByType(int type);
	int getNumberOfOutputsByType(int type);
	framemetadata_sp getOutputMetadataByType(int type);
	bool isMetadataEmpty(framemetadata_sp& metadata);
	bool isFrameEmpty(frame_sp& frame);
	string getInputPinIdByType(int type);
	string getOutputPinIdByType(int type);		
	
	bool setNext(boost::shared_ptr<Module> next, bool open, bool isFeedback); // take all the output pins			
	bool setNext(boost::shared_ptr<Module> next, vector<string>& pinIdArr, bool open, bool isFeedback); 
	void addInputPin(framemetadata_sp& metadata, string& pinId, bool isFeedback); 
	virtual void addInputPin(framemetadata_sp& metadata, string& pinId); // throws exception if validation fails	
	boost::shared_ptr<FrameContainerQueue> getQue() { return mQue; }
	
	bool getPlayState() { return mPlay; }

	// only for unit test
	Connections getConnections() { return mConnections; } 	
	
	//following is useful for testing to know whats in queue
	frame_container try_pop();
	frame_container pop();

	bool processSourceQue();
	bool handlePausePlay(bool play);
	virtual void notifyPlay(bool play) {}
private:	
	bool push(frame_container frameContainer); //exchanges the buffer 
	bool try_push(frame_container frameContainer); //tries to exchange the buffer
	
	bool addEoPFrame(frame_container& frames);
	bool handleStop();

	template<class T>
	void serialize(T& props, frame_sp& frame)
	{
		boost::iostreams::basic_array_sink<char> device_sink((char*)frame->data(), frame->size());
		boost::iostreams::stream<boost::iostreams::basic_array_sink<char> > s_sink(device_sink);

		boost::archive::binary_oarchive oa(s_sink);
		oa << props;
	}

	template<class T>
	void deSerialize(T& props, frame_sp& frame)
	{
		boost::iostreams::basic_array_source<char> device((char*)frame->data(), frame->size());
		boost::iostreams::stream<boost::iostreams::basic_array_source<char> > s(device);
		boost::archive::binary_iarchive ia(s);

		ia >> props;
	}

	bool shouldForceStep();
	bool shouldSkip();

	bool isFeedbackEnabled(std::string& moduleId); // get pins and call
	
	bool mPlay;
	uint32_t mForceStepCount;
	int mSkipIndex;
	Kind myNature;
	string myName;
	string myId;
	boost::thread myThread;
	boost::shared_ptr<FrameContainerQueue> mQue;
	bool mRunning;
	uint32_t mStopCount;
	uint32_t mForwardPins;
	boost::shared_ptr<FrameFactory> mpFrameFactory;
	boost::shared_ptr<FrameFactory> mpCommandFactory;
	boost::shared_ptr<PaceMaker> pacer;
	
	Connections mConnections; // For each module, all the required pins
	map<string, boost::shared_ptr<Module>> mModules;
	map<string, bool> mRelay;
		
	std::map<std::string, bool> mInputPinsDirection;
	metadata_by_pin mInputPinIdMetadataMap;
	metadata_by_pin mOutputPinIdMetadataMap;
	std::shared_ptr<FIndexStrategy> mFIndexStrategy;

	class Profiler;
	boost::shared_ptr<Profiler> mProfiler;
	boost::shared_ptr<ModuleProps> mProps;
	boost::shared_ptr<QuePushStrategy> mQuePushStrategy;

	framemetadata_sp mCommandMetadata;
	framemetadata_sp mPropsChangeMetadata;
};