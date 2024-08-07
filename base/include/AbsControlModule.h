#pragma once
#include "APCallback.h"
#include "Command.h"
#include "Module.h"
#include <map>

class PipeLine;
class AbsControlModuleProps : public ModuleProps {
public:
	AbsControlModuleProps() {}
};

class AbsControlModule : public Module {
public:
	AbsControlModule(AbsControlModuleProps _props);
	~AbsControlModule();
	bool init();
	bool term();
	bool enrollModule(std::string role, boost::shared_ptr<Module> module);
	boost::shared_ptr<Module> getModuleofRole(std::string role);
	virtual void handleMp4MissingVideotrack(std::string previousVideoFile, std::string nextVideoFile) {}
	virtual void handleMMQExport(Command cmd, bool priority = false) {}
	virtual void handleMMQExportView(uint64_t startTS, uint64_t endTS = 9999999999999, bool playabckDirection = true, bool Mp4ReaderExport = false, bool priority = false) {}
	virtual void handleSendMMQTSCmd(uint64_t  mmqBeginTS, uint64_t mmqEndTS, bool priority = false) {}
	virtual void handleLastGtkGLRenderTS(uint64_t  latestGtkGlRenderTS, bool priority) {}
	virtual void handleGoLive(bool goLive, bool priority) {}
	virtual void handleDecoderSpeed(DecoderPlaybackSpeed cmd, bool priority) {}
	boost::container::deque<boost::shared_ptr<Module>> pipelineModules;
	std::map<std::string, boost::shared_ptr<Module>> moduleRoles;
  	virtual void handleError(const APErrorObject &error) {}
  	virtual void handleHealthCallback(const APHealthObject &healthObj) {}


protected:
	bool process(frame_container& frames);
	bool handleCommand(Command::CommandType type, frame_sp& frame);
	bool handlePropsChange(frame_sp& frame);
	virtual void sendEOS() {}
	virtual void sendEOS(frame_sp& frame) {}
	virtual void sendEOPFrame() {}

private:
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};