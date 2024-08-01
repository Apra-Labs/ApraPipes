#include "APErrorObject.h"
#include "Logger.h"
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>

APErrorObject::APErrorObject(int errCode, const std::string &errorMsg)
    : mErrorCode(errCode), mErrorMessage(errorMsg), mModuleName(""),
      mModuleId("")
{
  mTimestamp = getCurrentTimestamp();
}

int APErrorObject::getErrorCode() const { return mErrorCode; }

std::string APErrorObject::getCurrentTimestamp() const
{
  auto now = std::chrono::system_clock::now();
  std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  std::tm tm = *std::localtime(&now_time);
  std::stringstream ss;
  ss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
  return ss.str();
}

std::string APErrorObject::getErrorMessage() const { return mErrorMessage; }

std::string APErrorObject::getModuleName() const { return mModuleName; }

std::string APErrorObject::getModuleId() const { return mModuleId; }

std::string APErrorObject::getTimestamp() const { return mTimestamp; }

void APErrorObject::displayError() const
{
  LOG_ERROR << "Module Name < " << mModuleName << " > Module Id < " << mModuleId
            << " > Time Stamp < " << mTimestamp << " > Error Message < "
            << mErrorMessage << " >";
}

void APErrorObject::setErrorCode(int errCode) 
{ 
  mErrorCode = errCode; 
}

void APErrorObject::setErrorMessage(const std::string &errorMsg)
{
  mErrorMessage = errorMsg;
}

void APErrorObject::setModuleName(const std::string &modName)
{
  mModuleName = modName;
}

void APErrorObject::setModuleId(const std::string &modId) { mModuleId = modId; }
