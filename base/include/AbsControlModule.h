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
	std::string printStatus();
	virtual void handleMp4MissingVideotrack(std::string previousVideoFile, std::string nextVideoFile) {}
	virtual void handleMMQExport(Command cmd, bool priority = false) {}
	virtual void handleMMQExportView(uint64_t startTS, uint64_t endTS = 9999999999999, bool playabckDirection = true, bool Mp4ReaderExport = false, bool priority = false) {}
	virtual void handleSendMMQTSCmd(uint64_t  mmqBeginTS, uint64_t mmqEndTS, bool priority = false) {}
	virtual void handleLastGtkGLRenderTS(uint64_t  latestGtkGlRenderTS, bool priority) {}
	virtual void handleGoLive(bool goLive, bool priority) {}
	virtual void handleDecoderSpeed(DecoderPlaybackSpeed cmd, bool priority) {}
	// Note: weak pointers to avoid cyclic dependency and mem leaks
	std::map<std::string, boost::weak_ptr<Module>> moduleRoles;
  	virtual void handleError(const APErrorObject &error) {}
	virtual void handleHealthCallback(const APHealthObject& healthObj);
	/**
	 * @brief Register external function to be triggered on every health callBack that control modules recieves from the modules.
	 * For eg. In SimpleControlModule, this extention is called at the end of handleHealthCallback function.
	 * @param function with signature void f(const APHealthObject*, unsigned short)
	 * @return nothing.
	 */
	void registerHealthCallbackExtention(
		boost::function<void(const APHealthObject*, unsigned short)> callbackFunction);
protected:
	bool process(frame_container& frames);
	bool handleCommand(Command::CommandType type, frame_sp& frame);
	bool handlePropsChange(frame_sp& frame);
	virtual void sendEOS() {}
	virtual void sendEOS(frame_sp& frame) {}
	virtual void sendEOPFrame() {}
	std::vector<std::string> serializeControlModule();
	boost::function<void(const APHealthObject*, unsigned short)> healthCallbackExtention;
private:
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};