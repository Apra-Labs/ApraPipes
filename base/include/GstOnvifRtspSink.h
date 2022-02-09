#pragma once

#include "Module.h"
#include <boost/serialization/vector.hpp>

class GStreamerOnvifRTSPSinkProps : public ModuleProps
{
public:
	struct User
	{
		std::string username;
		std::string password;

		size_t getSerializeSize()
		{
			return sizeof(username) + sizeof(password) + username.length() + password.length();
		}

	private:
		friend class boost::serialization::access;

		template <class Archive>
		void serialize(Archive &ar, const unsigned int version)
		{
			ar &username &password;
		}
	};
	GStreamerOnvifRTSPSinkProps(uint32_t _width = 1920, uint32_t _height = 1080, int _bitrate = 204800, int _goplength = 30, std::string _h264Profile = "baseline", std::string _port = "8554", std::string _htdigestPath = "/var/lib/onvif/onvif_users.htdigest", std::string _realm = "Onvif_service", std::string _unicastAddress = "0.0.0.0", std::string _mountPoint = "/rgbcamera", std::vector<User> _userList = {{"admin", "7ABCDE"}}) : ModuleProps(), width(_width), height(_height), bitrate(_bitrate), goplength(_goplength), h264Profile(_h264Profile), port(_port), htdigestPath(_htdigestPath), unicastAddress(_unicastAddress), realm(_realm), mountPoint(_mountPoint), userList(_userList) {}
	// port, authpaths, rtspLinkInitial
	uint32_t width;
	uint32_t height;
	std::vector<User> userList;
	int bitrate;
	int goplength;
	std::string h264Profile;
	std::string port;
	std::string htdigestPath;
	std::string realm;
	std::string unicastAddress;
	std::string mountPoint;
	size_t getSerializeSize()
	{
		size_t totalUserListSize = 0;

		for (User &user : userList)
		{
			totalUserListSize += user.getSerializeSize();
		}
		return totalUserListSize + ModuleProps::getSerializeSize() + sizeof(width) + sizeof(height) + sizeof(userList) + sizeof(bitrate) + sizeof(goplength) + sizeof(h264Profile) + h264Profile.length() + sizeof(port) + port.length() + sizeof(htdigestPath) + htdigestPath.length() + sizeof(realm) + realm.length() + sizeof(unicastAddress) + unicastAddress.length() + sizeof(mountPoint) + mountPoint.length();
	}

private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar &boost::serialization::base_object<ModuleProps>(*this);
		ar &width &height &userList &bitrate &goplength &h264Profile &port &htdigestPath &realm &unicastAddress &mountPoint;
	}
};

class GStreamerOnvifRTSPSink : public Module
{

public:
	GStreamerOnvifRTSPSink(GStreamerOnvifRTSPSinkProps props);
	virtual ~GStreamerOnvifRTSPSink();
	bool init();
	bool term();
	class Detail;
	boost::shared_ptr<Detail> mDetail;
	void setProps(GStreamerOnvifRTSPSinkProps &props);
	GStreamerOnvifRTSPSinkProps getProps();

protected:
	bool process(frame_container &frames);
	bool processSOS(frame_sp &frame);
	bool validateInputPins();
	bool shouldTriggerSOS();
	bool processEOS(string &pinId);
	bool handlePropsChange(frame_sp &frame);
};