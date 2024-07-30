#pragma once
#include <string>

class APHealthObject
{
private:
  std::string mModuleId;
  std::string mTimestamp;

  std::string getCurrentTimestamp() const;

public:
  APHealthObject(const std::string &modId);

  std::string getModuleId() const;
  std::string getTimestamp() const;

  void setModuleId(const std::string &modId);
};
