#include "FIndexStrategy.h"

std::shared_ptr<FIndexStrategy> FIndexStrategy::create(FIndexStrategyType type)
{
    switch (type)
    {
    case FIndexStrategyType::AUTO_INCREMENT:
        return std::make_shared<FIndexAutoIncrementStrategy>();
    case FIndexStrategyType::NONE:
        return std::make_shared<FIndexStrategy>();
        break;
    }
}

FIndexStrategy::FIndexStrategy()
{
}

FIndexStrategy::~FIndexStrategy()
{
}

uint64_t FIndexStrategy::getFIndex(uint64_t fIndex)
{
    return fIndex;
}

FIndexAutoIncrementStrategy::FIndexAutoIncrementStrategy() : FIndexStrategy(), mFIndex(0)
{
}

FIndexAutoIncrementStrategy::~FIndexAutoIncrementStrategy()
{
}

uint64_t FIndexAutoIncrementStrategy::getFIndex(uint64_t fIndex)
{
    if (fIndex == 0 || fIndex <= mFIndex)
    {
        // fIndex is being set (incremented) automatically
        fIndex = mFIndex++;
    }
    else
    {
        mFIndex = fIndex;
        mFIndex++;
    }

    return fIndex;
}