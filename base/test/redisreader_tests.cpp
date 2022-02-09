#include <boost/test/unit_test.hpp>
#include <sw/redis++/redis++.h>

#include "Logger.h"
#include "AIPExceptions.h"
#include "RedisDBReader.h"
#include "StatSink.h"
#include <cstdlib>

#define MAP_STRING_TO_BOOL(str, var) ( \
    std::istringstream(map.find(str)->second) >> std::boolalpha >> var)

#define MAP_STRING_TO_INT(str, var) ( \
    var = atoi(map.find(str)->second.c_str()))

#define MAP_STRING_TO_STRING(str, var) ( \
    var = map.find(str)->second)

BOOST_AUTO_TEST_SUITE(redisreader_tests)

BOOST_AUTO_TEST_CASE(readertest, *boost::unit_test::disabled())
{
    struct VideoEncoder
    {
        bool multicastautostart;
        std::string token;
        int multicastTTL;
        int bitrate;
        int framerate;
        std::string multicastIPAddress;
        bool multicastEnable;
        std::string encoding;
        int govlength;
        std::string name;
        int multicastPort;
        VideoEncoder(boost::shared_ptr<RedisDBReader> redisReader, sw::redis::Redis &redis, std::string videoEncoderName)
        {
            update(redisReader, redis, videoEncoderName);
        }

        void update(boost::shared_ptr<RedisDBReader> redisReader, sw::redis::Redis &redis, std::string videoEncoderName)
        {
            char hashName[1000];

            sprintf(hashName, "onvif.media.VideoEncoder#%s", videoEncoderName.c_str());
            auto map = redisReader->getHash(redis, hashName);

            MAP_STRING_TO_BOOL("multicastautostart", multicastautostart);

            MAP_STRING_TO_STRING("token", token);

            MAP_STRING_TO_INT("multicastTTL", multicastTTL);

            MAP_STRING_TO_INT("bitrate", bitrate);

            MAP_STRING_TO_INT("framerate", framerate);

            MAP_STRING_TO_STRING("multicastIPAddress", multicastIPAddress);

            MAP_STRING_TO_BOOL("multicastEnable", multicastEnable);

            MAP_STRING_TO_STRING("encoding", encoding);

            MAP_STRING_TO_INT("govlength", govlength);

            MAP_STRING_TO_STRING("name", name);

            MAP_STRING_TO_INT("multicastPort", multicastPort);
        }
    };

    struct Profile
    {
        bool fixed;
        std::string videoEncoderToken;
        std::string videoSourceToken;
        std::string name;
        std::string token;
        Profile(boost::shared_ptr<RedisDBReader> redisReader, sw::redis::Redis &redis, std::string profileName)
        {
            update(redisReader, redis, profileName);
        }

        void update(boost::shared_ptr<RedisDBReader> redisReader, sw::redis::Redis &redis, std::string profileName)
        {
            char hashName[1000];
            sprintf(hashName, "onvif.media.Profiles#%s", profileName.c_str());
            auto map = redisReader->getHash(redis, hashName);

            MAP_STRING_TO_BOOL("fixed", fixed);

            MAP_STRING_TO_STRING("videoEncoderToken", videoEncoderToken);

            MAP_STRING_TO_STRING("videoSourceToken", videoSourceToken);

            MAP_STRING_TO_STRING("name", name);

            MAP_STRING_TO_STRING("token", token);
        }
    };
    //Redis DB needs to be populated -  prerequisite
    LoggerProps logprops;
    logprops.logLevel = boost::log::trivial::severity_level::info;
    Logger::initLogger(logprops);

    sw::redis::ConnectionOptions connection_options;
    connection_options.type = sw::redis::ConnectionType::UNIX;
    connection_options.path = "/run/redis/redis.sock";
    connection_options.db = 1;
    connection_options.socket_timeout = std::chrono::milliseconds(1000);
    sw::redis::Redis redis = sw::redis::Redis(connection_options);

    // auto redisInstance = boost::shared_ptr<RedisRepositoryController>(new RedisRepositoryController());
    auto redisReader = boost::shared_ptr<RedisDBReader>(new RedisDBReader());
    auto userList = redisReader->getUsersList(redis);
    auto videoEncoderRGBCameraH264 = VideoEncoder(redisReader, redis, "rgbcamera_H264");
    LOG_INFO << videoEncoderRGBCameraH264.bitrate;
    auto profileFixed0 = Profile(redisReader, redis, "Fixed_0");
    LOG_INFO << profileFixed0.videoEncoderToken;
    auto profiles = redisReader->getSetMembers(redis, "onvif.media.Profiles");
    LOG_INFO << "userList Fetched";
}

BOOST_AUTO_TEST_SUITE_END()