#pragma once
#include "Module.h"
#include "AbsControlModule.h"
#include "NVRHelper.h"
class NVRControlModuleProps : public AbsControlModuleProps
{
public:
	NVRControlModuleProps()
	{
	}
	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize();
	}
private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive& ar, const unsigned int version)
	{
		ar& boost::serialization::base_object<ModuleProps>(*this);
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
	bool record(bool record);
	bool export(uint64_t startTime, uint64_t stopTime);

protected:
	bool validateInputPins();
	bool validateOutputPins();
	bool validateInputOutputPins();
	bool handleCommand(Command::CommandType type, frame_sp& frame);
	bool handlePropsChange(frame_sp& frame);

private:
	void setMetadata(framemetadata_sp& metadata);
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};