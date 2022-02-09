#include "EventSource.h"
#include <sw/redis++/redis++.h>

EventSource::EventSource()
{
}

EventSource::~EventSource()
{
    subscriber_running = false;
}

void EventSource::callbackWatcher(sw::redis::Redis& redis)
{

    auto subscriber = redis.subscriber();
    subscriber.on_pmessage([&](std::string pattern, std::string channel, std::string msg)
                           {
                               onChange(pattern);
                           });

    subscriber.psubscribe({"__keyspace@1__:onvif.media.Profiles*",
                                                                  "__keyspace@1__:onvif.media.VideoEncoder*",
                                                                  "__keyspace@1__:onvif.media.MetadataConfiguration*",
                                                                  "__keyspace@1__:onvif.users.User*",
                                                                  "__keyspace@1__:onvif.media.OSDConfigurations*",
                                                                  "__keyspace@1__:onvif.media.VideoSourceConfiguration*"});
    subscriber_running = true;
    while (subscriber_running)
    {
        try
        {
            subscriber.consume();
        }
        catch (const sw::redis::TimeoutError &e)
        {
            continue;
        }
        catch (const sw::redis::Error &err)
        {
            std::cout << "caught some other err" << std::endl;
            // Handle other exceptions.
        }
    }
}

void EventSource::listenKey(std::string key, std::function<void()> callback)
{
    listenKeys.emplace(key, callback);
    return;
}

void EventSource::onChange(std::string key)
{
    auto callbackIterator = listenKeys.find(key);
    callbackIterator->second();
    return;
}