#pragma once

#include "Module.h"
#include <vector>
#include <boost/shared_ptr.hpp>

class TeeModuleProps : public ModuleProps {
public:
    TeeModuleProps() {}
};

class TeeModule : public Module {
public:
    TeeModule(const TeeModuleProps& props = TeeModuleProps());
    virtual ~TeeModule();

    // Add a downstream module to tee frames to
    bool enrollModule(const std::string& role, boost::shared_ptr<Module> module);

    // Get enrolled module by role
    boost::shared_ptr<Module> getModuleByRole(const std::string& role);

    // Override process - send frame to all enrolled modules
    virtual bool process(frame_container& frame) override;

    // Override init/term if needed
    virtual bool init() override;
    virtual bool term() override;

protected:
    std::map<std::string, boost::weak_ptr<Module>> mDownstreamModules;
};
