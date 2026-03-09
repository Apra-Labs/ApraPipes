#pragma once

#include <vector>
#include <memory>
#include "Module.h"

class ModulePool : public Module {
public:
    ModulePool(const Props& props, int initialSize, int maxSize);
    ~ModulePool();

    void init() override;
    void term() override;
    void setProp(const Props& props) override;
    void process(Frame& frame) override;

private:
    std::vector<std::shared_ptr<Module>> moduleInstances;
    int currentIndex;
};