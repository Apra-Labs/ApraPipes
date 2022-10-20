#pragma once
#include "Module.h"
#include "AbsControlModule.h"

class NVRControlModuleProps : public AbsControlModuleProps
{
public:
	NVRControlModuleProps()
	{
	}
};

class NVRControlModule : public AbsControlModule
{
	public:
	NVRControlModule(NVRControlModuleProps _props);
	~NVRControlModule();
	bool init();
	bool term();
	void setProps(NVRControlModuleProps& props);
	NVRControlModuleProps getProps();

protected:
	bool process(frame_container& frames);
	bool validateInputPins();
	bool validateOutputPins();
	bool validateInputOutputPins();
	void addInputPin(framemetadata_sp& metadata, string& pinId);
	bool handleCommand(Command::CommandType type, frame_sp& frame);
	bool handlePropsChange(frame_sp& frame);
	bool Record(bool record);
	bool Export(uint64_t startTime,uint64_t stopTime);

private:
	void setMetadata(framemetadata_sp& metadata);
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};