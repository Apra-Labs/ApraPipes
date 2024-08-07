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

	void handleError(const APErrorObject &error);
    void handleHealthCallback(const APHealthObject &healthObj);

	// ErrorCallbacks
protected:
	void sendEOS();
	void sendEOS(frame_sp& frame);
	void sendEOPFrame();
private:
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};
