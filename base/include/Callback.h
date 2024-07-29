#pragma once
#include "ErrorObject.h"
#include "HealthObject.h"
#include <functional>

using ErrorCallback = std::function<void(const ErrorObject &)>;
using HealthCallback = std::function<void(const HealthObject &)>;
