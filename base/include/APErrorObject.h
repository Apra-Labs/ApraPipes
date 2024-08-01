#pragma once
#include <string>

class APErrorObject {
private:
  int mErrorCode;
  std::string mErrorMessage;
  std::string mModuleName;
  std::string mModuleId;
  std::string mTimestamp;

  std::string getCurrentTimestamp() const;

public:
  APErrorObject(int errCode, const std::string &errorMsg);

  int getErrorCode() const;
  std::string getErrorMessage() const;
  std::string getModuleName() const;
  std::string getModuleId() const;
  std::string getTimestamp() const;

  void displayError() const;
  void setErrorCode(int errCode);
  void setErrorMessage(const std::string &errorMsg);
  void setModuleName(const std::string &modName);
  void setModuleId(const std::string &modId);
};
