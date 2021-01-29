#pragma once

#include <cstdint>
#include <memory>

class FIndexStrategy
{
public:
    enum FIndexStrategyType {		
		AUTO_INCREMENT,
		NONE
	};

    static std::shared_ptr<FIndexStrategy> create(FIndexStrategyType type);

    FIndexStrategy();
    virtual ~FIndexStrategy();

    virtual uint64_t getFIndex(uint64_t fIndex);
};

class FIndexAutoIncrementStrategy: public FIndexStrategy
{
public:
    FIndexAutoIncrementStrategy();
    ~FIndexAutoIncrementStrategy();

    uint64_t getFIndex(uint64_t fIndex);

private:
	uint64_t mFIndex;
};