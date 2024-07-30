#pragma once
#include "APErrorObject.h"
#include "APHealthObject.h"
#include <functional>

using APErrorCallback = std::function<void(const APErrorObject &)>;
using APHealthCallback = std::function<void(const APHealthObject &)>;
