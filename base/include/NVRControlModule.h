#pragma once
#include "Module.h"
#include "AbsControlModule.h"

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
	bool validateModuleRoles();
	bool nvrRecord(bool record);
	bool nvrExport(uint64_t startTime, uint64_t stopTime);
	bool nvrExportView(uint64_t startTime, uint64_t stopTime);
	bool nvrView(bool view);
	bool nvrGoLive();
	bool isRendererPaused = false;
	uint64_t pausedTS = 0;
	uint64_t mp4lastWrittenTS = 0;
	uint64_t firstMMQtimestamp = 0;
	uint64_t lastMMQtimestamp = 0;
	uint64_t givenStart = 0;
	uint64_t givenStop = 0;
	uint64_t mp4_2_lastWrittenTS = 0;
	bool isExporting = false;
	bool isRenderWindowOpen = true;
	bool isStateLive = true;
	uint64_t currentRenderTS = 0;
	bool isSavingVideo = false;
    bool isExpWriterInitialized = true;

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