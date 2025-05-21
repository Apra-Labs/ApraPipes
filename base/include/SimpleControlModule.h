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
	std::string printStatus();
	void handleError(const APErrorObject& error) override;
	void handleHealthCallback(const APHealthObject& healthObj) override;
protected:
	void sendEOS();
	void sendEOS(frame_sp& frame);
	void sendEOPFrame();
private:
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};
