#pragma once
#include "APErrorObject.h"
#include "APHealthObject.h"
#include <functional>

using ErrorCallback = std::function<void(const APErrorObject &)>;
using HealthCallback = std::function<void(const APHealthObject &)>;
