#include "APHealthObject.h"
#include "Logger.h"
#include <ctime>
#include <chrono>

APHealthObject::APHealthObject(const std::string &modId) : mModuleId(modId)
{
  mTimestamp = getCurrentTimestamp();
}

std::string APHealthObject::getCurrentTimestamp() const
{
  auto now = std::chrono::system_clock::now();
  std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  std::tm tm = *std::localtime(&now_time);
  std::stringstream ss;
  ss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
  return ss.str();
}

std::string APHealthObject::getModuleId() const { return mModuleId; }

std::string APHealthObject::getTimestamp() const { return mTimestamp; }

void APHealthObject::setModuleId(const std::string &modId) { mModuleId = modId; }
