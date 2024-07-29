#include "HealthObject.h"
#include "Logger.h"
#include <ctime>
#include <iostream>

HealthObject::HealthObject(const std::string &modId) : mModuleId(modId) {
  mTimestamp = getCurrentTimestamp();
}

std::string HealthObject::getCurrentTimestamp() const {
  // Implement a function to get the current timestamp
  // For now, returning a placeholder
  return "2024-07-12 10:00:00";
}

std::string HealthObject::getModuleId() const { return mModuleId; }

std::string HealthObject::getTimestamp() const { return mTimestamp; }

void HealthObject::setModuleId(const std::string &modId) { mModuleId = modId; }
