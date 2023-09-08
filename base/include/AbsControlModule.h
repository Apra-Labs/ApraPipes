#pragma once
#include <map>
#include "Module.h"

class AbsControlModuleProps : public ModuleProps
{
public:
	AbsControlModuleProps()
	{
	}
};

class AbsControlModule : public Module
{
public:
	AbsControlModule(AbsControlModuleProps _props);
	~AbsControlModule();
	bool init();
	bool term();
	void setProps(AbsControlModuleProps& props);
	bool enrollModule(std::string role, boost::shared_ptr<Module> module);
	boost::shared_ptr<Module> getModuleofRole(std::string role);
	AbsControlModuleProps getProps();
	boost::container::deque<boost::shared_ptr<Module>> pipelineModules;
	std::map<std::string, boost::shared_ptr<Module>> moduleRoles;

protected:
	bool process(frame_container& frames);
	bool validateInputPins();
	bool validateOutputPins();
	bool validateInputOutputPins();
	void addInputPin(framemetadata_sp& metadata, string& pinId);
	bool handleCommand(Command::CommandType type, frame_sp& frame);
	bool handlePropsChange(frame_sp& frame);

private:
	void setMetadata(framemetadata_sp& metadata);
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};