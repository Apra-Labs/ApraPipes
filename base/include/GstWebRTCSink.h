#pragma once

#include "Module.h"

class GStreamerWebRTCSinkProps : public ModuleProps
{
public:
	GStreamerWebRTCSinkProps(uint32_t _width = 640, uint32_t _height = 480, int _bitrate = 204800, int _goplength = 1, std::string _h264Profile = "high", std::string _peerId = "" , std::string _signallingSrvEndpoint="") : ModuleProps(), width(_width), height(_height), bitrate(_bitrate), goplength(_goplength), h264Profile(_h264Profile), peerId(_peerId), signallingSrvEndpoint(_signallingSrvEndpoint) {}
	
	uint32_t width;
	uint32_t height;
	int bitrate;
	int goplength;
	std::string h264Profile;
	std::string peerId;
	std::string signallingSrvEndpoint;
	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(width) + sizeof(height) + sizeof(bitrate) + sizeof(goplength) + sizeof(h264Profile) + h264Profile.length() + sizeof(peerId) + peerId.length() + sizeof(signallingSrvEndpoint) + signallingSrvEndpoint.length();
	}

private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar &boost::serialization::base_object<ModuleProps>(*this);
		ar &width &height &bitrate &goplength &h264Profile &peerId &signallingSrvEndpoint;
	}
};

class GStreamerWebRTCSink : public Module
{

public:
	GStreamerWebRTCSink(GStreamerWebRTCSinkProps props);
	virtual ~GStreamerWebRTCSink();
	bool init();
	bool term();
	class Detail;
	boost::shared_ptr<Detail> mDetail;
	void setProps(GStreamerWebRTCSinkProps &props);
	GStreamerWebRTCSinkProps getProps();

protected:
	bool process(frame_container &frames);
	bool processSOS(frame_sp &frame);
	bool validateInputPins();
	bool shouldTriggerSOS();
	bool processEOS(string &pinId);
	bool handlePropsChange(frame_sp &frame);
};