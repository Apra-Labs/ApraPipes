#include "stdafx.h"
#include "PipeLine.h"
#include "Module.h"





PipeLine::~PipeLine()
{ 
	//LOG_INFO << "Dest'r ~PipeLine" << mName << endl;
}


bool PipeLine::appendModule(boost::shared_ptr<Module> pModule)
{	
	// assumes that appendModule is called after all the connections

	//remember all the modules
	auto pos = std::find(modules.begin(), modules.end(), pModule);
	if (pos != modules.end())
	{		
		// already added
		return true;
	}

	modules.push_back(pModule);	
	auto nextModules = pModule->getConnectedModules(); // get next modules
	for (auto i = nextModules.begin(); i != nextModules.end(); i++)
	{
		// recursive call
		appendModule(*i);
	}	
	
	return true;
}

bool PipeLine::checkCyclicDependency()
{
	std::map< std::string, std::vector<std::string> > dependencyMap;
	auto valid = true;

	for (auto itr = modules.begin(); itr != modules.end(); itr++)
	{
		auto parentModule = itr->get();
		auto parentId = parentModule->getId();

		dependencyMap[parentId] = std::vector<std::string>();
		auto nextModules = parentModule->getConnectedModules(); // get next modules
		for (auto childItr = nextModules.begin(); childItr != nextModules.end(); childItr++)
		{
			auto childModule = childItr->get();
			auto childId = childModule->getId();

			dependencyMap[parentId].push_back(childId);

			if (dependencyMap.find(childId) == dependencyMap.end())
			{
				dependencyMap[childId] = std::vector<std::string>();
			}

			for (const auto& moduleId: dependencyMap[childId])
			{
				if (moduleId == parentId)
				{
					// cyclic dependency
					if (!parentModule->isFeedbackEnabled(childId) && !childModule->isFeedbackEnabled(parentId))
					{
						// feedback not enabled
						LOG_ERROR << "Cyclic Dependency detected between <" << parentId << "> and <" << childId << ">. Please use addFeedback when connecting the pins for one of the module.";
						valid = false;
					}
				}
			}
		}
	}

	return valid;
}

bool PipeLine::validate()
{
	if (modules.front()->getNature() != Module::SOURCE)
	{
		LOG_ERROR << "Pipeline should start with a source, but starts with " << modules.front()->getId();
		return false;
	}
	if (modules.back()->getNature() != Module::SINK)
	{
		LOG_ERROR << "Pipeline should end with a sink, but ends with " << modules.back()->getId();
		return false;
	}
	return true;
}

bool PipeLine::init()
{
	if(!checkCyclicDependency())
	{
		myStatus = PL_INITFAILED;
		return false;
	}

	LOG_TRACE << " Initializing pipeline";
	for (auto i = modules.begin(); i != modules.end(); i++)
	{
		bool bRCInit = false;
		try {
			bRCInit = i->get()->init();
			LOG_TRACE << " Initialized pipeline" << i->get()->getId();
		}
		catch (const std::exception& ex)
		{
			LOG_ERROR << "Failed init " << i->get()->getId() << "Exception" << ex.what(); 
			
		}
		catch (...)
		{
			LOG_ERROR << "Failed init " << i->get()->getId() << "Unknown Exception";
			
		}
		if (!bRCInit)
		{
			LOG_ERROR << " Failed init " << i->get()->getId();
			myStatus = PL_INITFAILED;
			return false;
		}
	}
	myStatus = PL_INITED;
	LOG_TRACE << " Pipeline initialized";
	return true;
}


void PipeLine::term()
{
	if (myStatus >= PL_TERMINATING)
	{
		LOG_INFO << "Pipeline status " << getStatus() << " Can not be terminated !";
		return;
	}

	myStatus = PL_TERMINATED;
}

void PipeLine::run_all_threaded()
{
	myStatus = PL_RUNNING;
	for (auto i = modules.begin(); i != modules.end(); i++)
	{
		Module& m = *(i->get());
		m.myThread=boost::thread(ref(m));
	}

	mPlay = true;
}

void PipeLine::run_all_threaded_withpause()
{
	pause();
	run_all_threaded();
	mPlay = false;
}

void PipeLine::pause()
{
	for (auto i = modules.begin(); i != modules.end(); i++)
	{
		if (i->get()->getNature() == Module::SOURCE)
		{
			i->get()->play(false);
		}
	}

	mPlay = false;
}

void PipeLine::play()
{
	for (auto i = modules.begin(); i != modules.end(); i++)
	{
		if (i->get()->getNature() == Module::SOURCE)
		{
			i->get()->play(true);
		}
	}

	mPlay = true;
}

void PipeLine::step()
{
	if (mPlay)
	{
		// already playing
		return;
	}

	for (auto i = modules.begin(); i != modules.end(); i++)
	{
		if (i->get()->getNature() == Module::SOURCE)
		{
			i->get()->queueStep();
		}
	}
}

void PipeLine::stop()
{
	if (myStatus >= PL_STOPPING)
	{
		LOG_INFO << "Pipeline status " << getStatus() << " Can not be stopped !";
		return;
	}
	myStatus = PL_STOPPING;
	for (auto i = modules.begin(); i != modules.end(); i++)
	{
		if (i->get()->getNature() == Module::SOURCE)
		{
			i->get()->stop();
		}
	}
}

void PipeLine::wait_for_all()
{
	if (myStatus != PL_TERMINATED)
	{
		LOG_INFO << "Pipeline status " << getStatus() << " is expected to be PL_TERMINATED";
		return;
	}

	for (auto i = modules.begin(); i != modules.end(); i++)
	{
		Module& m = *(i->get());
		m.myThread.join();
	}
}


void PipeLine::interrup_wait_for_all()
{
	if (myStatus > PL_STOPPING)
	{
		LOG_INFO << "Pipeline status " << getStatus() << " Can not be stopped !";
		return;
	}

	for (auto i = modules.begin(); i != modules.end(); i++)
	{
		Module& m = *(i->get());
		m.myThread.interrupt();
	}

	for (auto i = modules.begin(); i != modules.end(); i++)
	{
		Module& m = *(i->get());
		m.myThread.join();
	}
	myStatus = PL_STOPPED;
}

const char * PipeLine::getStatus()
{
	const char* const StatusNames[] = {
		Status_ENUM(MAKE_STRINGS,X)
	};
	return StatusNames[myStatus];
}


