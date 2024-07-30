#pragma once
#include "AbsControlModule.h"

class SimpleControlModuleProps : public AbsControlModuleProps {
public:
	SimpleControlModuleProps() {}
};

class SimpleControlModule : public AbsControlModule
{
public:
	SimpleControlModule(SimpleControlModuleProps _props) : AbsControlModule(_props)
	{
	}

	~SimpleControlModule()
	{

	}

	void handleError(const ErrorObject &error);
    void handleHealthCallback(const HealthObject &healthObj);

	// ErrorCallbacks
protected:
	void sendEOS();
	void sendEOS(frame_sp& frame);
	void sendEOPFrame();
private:
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};
