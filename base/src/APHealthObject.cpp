#include "APHealthObject.h"
#include "Logger.h"
#include <ctime>

APHealthObject::APHealthObject(const std::string &modId) : mModuleId(modId)
{
  mTimestamp = getCurrentTimestamp();
}

std::string APHealthObject::getCurrentTimestamp() const
{
  // Implement a function to get the current timestamp
  // For now, returning a placeholder
  return "2024-07-12 10:00:00";
}

std::string APHealthObject::getModuleId() const { return mModuleId; }

std::string APHealthObject::getTimestamp() const { return mTimestamp; }

void APHealthObject::setModuleId(const std::string &modId) { mModuleId = modId; }
