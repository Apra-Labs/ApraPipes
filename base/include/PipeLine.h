#pragma once
#include <boost/shared_ptr.hpp>
#include <boost/container/deque.hpp>
#include "enum_macros.h"

using namespace std;
class Module;
// a linear pipline
class PipeLine {
	#define Status_ENUM(DO,C) \
	DO(PL_CREATED,C) \
	DO(PL_INITED,C) \
	DO(PL_INITFAILED,C) \
	DO(PL_RUNNING,C) \
	DO(PL_RUNFAILED,C) \
	DO(PL_STOPPING,C) \
	DO(PL_STOPPED,C) \
	DO(PL_TERMINATING,C) \
	DO(PL_TERMINATED,C) 

	enum Status {
		Status_ENUM(MAKE_ENUM,X)
	};

	bool mPlay;
	Status myStatus;
	typedef boost::shared_ptr<Module> item_type;
	typedef boost::container::deque< item_type > container_type;
	
	string mName;
	container_type modules;
	bool validate();
	bool checkCyclicDependency();
public:
	PipeLine(string name) :mName(name), myStatus(PL_CREATED), mPlay(false) {}
	~PipeLine();
	string getName() { return mName; }
	bool appendModule(boost::shared_ptr<Module> pModule);
	bool init();
	void run_all_threaded();
	void run_all_threaded_withpause();
	void pause();
	void play();
	void step();
	void stop();
	void term();
	void wait_for_all();
	void interrup_wait_for_all();
	const char* getStatus();
};

