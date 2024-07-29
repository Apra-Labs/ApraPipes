#include "ErrorObject.h"
#include "Logger.h"
#include <ctime>
#include <iostream>

ErrorObject::ErrorObject(int errCode, const std::string &errorMsg,
                         const std::string &modName, const std::string &modId)
    : mErrorCode(errCode), mErrorMessage(errorMsg), mModuleName(modName),
      mModuleId(modId) {
  mTimestamp = getCurrentTimestamp();
}

int ErrorObject::getErrorCode() const { return mErrorCode; }

std::string ErrorObject::getCurrentTimestamp() const {
  // Implement a function to get the current timestamp
  // For now, returning a placeholder
  return "2024-07-12 10:00:00";
}

std::string ErrorObject::getErrorMessage() const { return mErrorMessage; }

std::string ErrorObject::getModuleName() const { return mModuleName; }

std::string ErrorObject::getModuleId() const { return mModuleId; }

std::string ErrorObject::getTimestamp() const { return mTimestamp; }

void ErrorObject::displayError() const {
  LOG_ERROR << "Module Name < " << mModuleName << " > Module Id < " << mModuleId
            << " > Time Stamp < " << mTimestamp << " > Error Message < "
            << mErrorMessage << " >";
}

void ErrorObject::setErrorCode(int errCode) {
  mErrorCode = errCode;
  mTimestamp = getCurrentTimestamp();
}

void ErrorObject::setErrorMessage(const std::string &errorMsg) {
  mErrorMessage = errorMsg;
  mTimestamp = getCurrentTimestamp();
}

void ErrorObject::setModuleName(const std::string &modName) {
  mModuleName = modName;
}

void ErrorObject::setModuleId(const std::string &modId) { mModuleId = modId; }
