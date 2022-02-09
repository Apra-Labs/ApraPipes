//read the key from db
//make properties and set props
//define some keys
//instantiate brightness class
//

#include <boost/test/unit_test.hpp>
#include <sw/redis++/redis++.h>
#include "BrightnessContrastControl.h"
#include "EventSource.h"
#include "Logger.h"
#include "AIPExceptions.h"
#include "RedisDBReader.h"
#include "StatSink.h"
#include "GstOnvifRtspSink.h"

BOOST_AUTO_TEST_SUITE(eventsource_tests)

BOOST_AUTO_TEST_CASE(eventsourcetest, *boost::unit_test::disabled())
{
    //Redis DB needs to be populated -  prerequisite
    LoggerProps logprops;
    logprops.logLevel = boost::log::trivial::severity_level::info;
    Logger::initLogger(logprops);

    sw::redis::ConnectionOptions connection_options;
    connection_options.type = sw::redis::ConnectionType::UNIX;
    connection_options.path = "/run/redis/redis.sock";
    connection_options.db = 1;
    connection_options.socket_timeout = std::chrono::milliseconds(10000);
    sw::redis::Redis redis = sw::redis::Redis(connection_options);

    auto redisReader = boost::shared_ptr<RedisDBReader>(new RedisDBReader());
    auto eventSource = boost::shared_ptr<EventSource>(new EventSource());
    GStreamerOnvifRTSPSinkProps props;
    auto rtspSink = boost::shared_ptr<GStreamerOnvifRTSPSink>(new GStreamerOnvifRTSPSink(props));

    eventSource->listenKey("__keyspace@1__:onvif.users.User*", [&]() -> void
                           {
                               auto userList = redisReader->getUsersList(redis);
                               if (!userList.empty())
                               {
                                   props.userList = userList;
                                   rtspSink->setProps(props);
                                   LOG_INFO << "userList Fetched on callback";
                               }
                           });

    eventSource->callbackWatcher(redis);
    boost::this_thread::sleep_for(boost::chrono::seconds(10000));
    LOG_INFO << "userList Fetched";
}

BOOST_AUTO_TEST_SUITE_END()