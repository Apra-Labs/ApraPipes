#pragma once

#include "Module.h"
#include <sw/redis++/redis++.h>


class EventSource
{
public:
	EventSource();
	virtual ~EventSource();
    void setPropsAfterReceivingCallback();
    void callbackWatcher(sw::redis::Redis& redis);
    void listenKey(std::string key, std::function<void()> callback);
    void onChange(std::string key);
    bool subscriber_running;
private:
    std::map<std::string,std::function<void()>> listenKeys;
};