#pragma once
#include <string>
#include "APErrorCodes.h"

class APErrorObject {
private:
  APErrorCodes mErrorCode;
  std::string mErrorMessage;
  std::string mModuleName;
  std::string mModuleId;
  std::string mTimestamp;

  std::string getCurrentTimestamp() const;

public:
  APErrorObject(APErrorCodes errCode, const std::string &errorMsg);

  APErrorCodes getErrorCode() const;
  std::string getErrorMessage() const;
  std::string getModuleName() const;
  std::string getModuleId() const;
  std::string getTimestamp() const;

  void displayError() const;

  void setErrorCode(APErrorCodes errCode);
  void setErrorMessage(const std::string &errorMsg);
  void setModuleName(const std::string &modName);
  void setModuleId(const std::string &modId);
};
