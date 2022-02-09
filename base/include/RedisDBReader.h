#include <sw/redis++/redis++.h>
#include "GstOnvifRtspSink.h"

class RedisDBReader
{
public:
    void *readValueFromVideoEncoderByField(sw::redis::Redis &redis, std::string encoderName, std::string fieldName);
    void *readImageSettingValue(sw::redis::Redis &redis, std::string imagingSetting);
    std::vector<GStreamerOnvifRTSPSinkProps::User> getUsersList(sw::redis::Redis &redis);
    std::vector<std::string> getSetMembers(sw::redis::Redis &redis, std::string setName);
    std::unordered_map<std::string, std::string> getHash(sw::redis::Redis &redis, std::string hashName);
    std::string getValueByKeyName(sw::redis::Redis &redis, std::string keyName);
};