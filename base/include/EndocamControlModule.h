#pragma once
#include "Module.h"
#include "AbsControlModule.h"

class EndocamControlModuleProps : public AbsControlModuleProps
{
public:
	EndocamControlModuleProps()
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

class EndocamControlModule : public AbsControlModule
{
	public:
	EndocamControlModule(EndocamControlModuleProps _props);
	~EndocamControlModule();
	bool init();
	bool term();
	bool validateModuleRoles();
	uint64_t mp4lastWrittenTS = 0;
	uint64_t firstMMQtimestamp = 0;
	uint64_t lastMMQtimestamp = 0;
	uint64_t givenStart = 0;
	uint64_t givenStop = 0;
	uint64_t mp4_2_lastWrittenTS = 0;
	bool isExporting = false;

protected:
	bool validateInputPins();
	bool validateOutputPins();
	bool validateInputOutputPins();
	bool handleCommand(Command::CommandType type, frame_sp& frame);

private:
	void setMetadata(framemetadata_sp& metadata);
};