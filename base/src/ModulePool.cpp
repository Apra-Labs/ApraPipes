#include "ModulePool.h"

ModulePool::ModulePool(const Props& props, int initialSize, int maxSize) : currentIndex(0) {
    for (int i = 0; i < initialSize; i++) {
        std::shared_ptr<Module> module = std::make_shared<Module>();
        module->setProp(props);
        moduleInstances.push_back(module);
    }
}

ModulePool::~ModulePool() {}

void ModulePool::init() {
    for (auto& module : moduleInstances) {
        module->init();
    }
}

void ModulePool::term() {
    for (auto& module : moduleInstances) {
        module->term();
    }
}

void ModulePool::setProp(const Props& props) {
    for (auto& module : moduleInstances) {
        module->setProp(props);
    }
}

void ModulePool::process(Frame& frame) {
    moduleInstances[currentIndex]->process(frame);
    currentIndex = (currentIndex + 1) % moduleInstances.size();
}