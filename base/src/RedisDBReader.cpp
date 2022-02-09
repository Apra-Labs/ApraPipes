#include "RedisDBReader.h"
#include <vector>
#include <string_view>
#include <string>
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include <iostream>
#include <sstream>
#include "AIPExceptions.h"
#include "GstOnvifRtspSink.h"


std::vector<GStreamerOnvifRTSPSinkProps::User> RedisDBReader::getUsersList(sw::redis::Redis &redis)
{
    std::unordered_set<std::string> user_keys;
    std::vector<GStreamerOnvifRTSPSinkProps::User> users;
    redis.smembers("onvif.users.User", std::inserter(user_keys, user_keys.begin()));
    for (const auto& user_key : user_keys) {
        std::unordered_map<std::string, std::string> userinfo;
        redis.hgetall(user_key, std::inserter(userinfo, userinfo.begin()));
        try {
            GStreamerOnvifRTSPSinkProps::User user = { userinfo.at("userName"), userinfo.at("password") };
            users.emplace_back(user);
            LOG_INFO << "Got username " << user.username;
        } catch (std::out_of_range &exc) {
            LOG_ERROR << "could not retrieve user information for " << user_key;
        }
    }
    return users;
}

std::unordered_map<std::string, std::string> RedisDBReader::getHash(sw::redis::Redis &redis, std::string hashName)
{
    std::unordered_map<std::string, std::string> hashdata;
    redis.hgetall(hashName,std::inserter(hashdata,hashdata.begin()));
    return hashdata;
}

std::string RedisDBReader::getValueByKeyName(sw::redis::Redis &redis, std::string keyName)
{
    LOG_INFO << "Fetching value for key named " << keyName;
    auto val = redis.get(static_cast<sw::redis::StringView>(keyName));
    return *val;
}

std::vector<std::string> RedisDBReader::getSetMembers(sw::redis::Redis &redis, std::string setName)
{
    std::vector<std::string> members;
    redis.smembers(static_cast<sw::redis::StringView>(setName),std::inserter(members,members.begin()));
    return members;
}