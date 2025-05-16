#pragma once
#include <chrono>
#include <string>
#include "Module.h"

class RTSPClientSrcProps : public ModuleProps
{
public:
	RTSPClientSrcProps(const std::string& rtspURL, const std::string& userName, const std::string& password, bool useTCP=true) : ModuleProps(),
		rtspURL(rtspURL), userName(userName), password(password),useTCP(useTCP)
	{
	}

	RTSPClientSrcProps()
	{
	}

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(rtspURL) + sizeof(userName) + sizeof(password), sizeof(useTCP);
	}

	string rtspURL, userName, password;
	bool useTCP;
private:
	friend class boost::serialization::access;

	template<class Archive>
	void serialize(Archive& ar, const unsigned int version)
	{
		ar& boost::serialization::base_object<ModuleProps>(*this);
		ar& rtspURL;
		ar& userName;
		ar& password;
		ar& useTCP;
	}
};

class RTSPClientSrc : public Module {
public:
	RTSPClientSrc(RTSPClientSrcProps _props);
	virtual ~RTSPClientSrc();
	bool init();
	bool term();
	void setProps(RTSPClientSrcProps& props);
	int getCurrentFps();
	RTSPClientSrcProps getProps();

protected:
	bool produce();
	bool validateOutputPins();
	void notifyPlay(bool play);
	bool handleCommand(Command::CommandType type, frame_sp& frame);
	bool handlePropsChange(frame_sp& frame);
private:
	RTSPClientSrcProps mProps;
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};
