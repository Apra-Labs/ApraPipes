#pragma once

#include "Module.h"

class WebCamSourceProps : public ModuleProps
{
public:
	WebCamSourceProps(int _cameraId = -1, uint32_t _width = 640, uint32_t _height = 480, int _fps = 30) : ModuleProps(), width(_width), height(_height), cameraId(_cameraId), fps(_fps) {}
	uint32_t width = 640;
	uint32_t height = 480;
	int cameraId = -1;
	int fps = 30;

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(width) + sizeof(height) + sizeof(cameraId) + sizeof(fps);
	}

private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar &boost::serialization::base_object<ModuleProps>(*this);
		ar &width &height &cameraId &fps;
	}
};

class WebCamSource : public Module
{
public:
	WebCamSource(WebCamSourceProps props);
	virtual ~WebCamSource();
	bool init();
	bool term();

	void setProps(WebCamSourceProps &props);
	WebCamSourceProps getProps();

protected:
	bool produce();
	bool validateOutputPins();
	bool handlePropsChange(frame_sp &frame);

private:
	class Detail;
	std::shared_ptr<Detail> mDetail;
};