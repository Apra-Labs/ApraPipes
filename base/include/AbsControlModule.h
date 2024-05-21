#pragma once
#include <map>
#include "Module.h"
#include "Command.h"

class PipeLine;
class AbsControlModuleProps : public ModuleProps
{
public:
	AbsControlModuleProps() {}
};

class AbsControlModule : public Module
{
public:
	AbsControlModule(AbsControlModuleProps _props);
	~AbsControlModule();
	bool init();
	bool term();
	std::string enrollModule(PipeLine p, std::string role, boost::shared_ptr<Module> module);
	std::pair<bool, boost::shared_ptr<Module>> getModuleofRole(PipeLine p, std::string role);
    virtual void handleMp4MissingVideotrack() {}
	virtual void handleMMQExport(Command cmd, bool priority = false) {}
	virtual void handleMMQExportView(Command cmd, bool priority = false) {}
	virtual void handleSendMMQTSCmd(SendMMQTimestamps cmd, bool priority = false) {}
	boost::container::deque<boost::shared_ptr<Module>> pipelineModules;
	std::map<std::string, boost::shared_ptr<Module>> moduleRoles;

protected:
	bool process(frame_container& frames);
	bool handleCommand(Command::CommandType type, frame_sp& frame);
	bool handlePropsChange(frame_sp& frame);

private:
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};