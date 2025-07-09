#include "TeeModule.h"
#include "Logger.h"

TeeModule::TeeModule(const TeeModuleProps& props)
    : Module("TeeModule", props)
{
}

TeeModule::~TeeModule() {}

bool TeeModule::enrollModule(const std::string& role, boost::shared_ptr<Module> module)
{
    if (mDownstreamModules.find(role) != mDownstreamModules.end())
    {
        LOG_ERROR << "Role '" << role << "' already enrolled in TeeModule.";
        return false;
    }
    mDownstreamModules[role] = module;
    return true;
}

boost::shared_ptr<Module> TeeModule::getModuleByRole(const std::string& role)
{
    boost::shared_ptr<Module> mod;
    try
    {
        mod = mDownstreamModules.at(role).lock();
    }
    catch (const std::out_of_range&)
    {
        LOG_ERROR << "No module found with role '" << role << "' in TeeModule.";
    }
    return mod;
}

bool TeeModule::process(frame_container& frames)
{
    for (auto& entry : mDownstreamModules)
    {
        auto mod = entry.second.lock();
        if (mod)
        {
            mod->process(frames);
        }
    }
    return true;
}


bool TeeModule::init()
{
    // Optionally initialize downstream modules here
    bool allOk = true;
    for (auto& entry : mDownstreamModules)
    {
        auto mod = entry.second.lock();
        if (mod)
        {
            allOk = allOk && mod->init();
        }
    }
    return allOk;
}

bool TeeModule::term()
{
    bool allOk = true;
    for (auto& entry : mDownstreamModules)
    {
        auto mod = entry.second.lock();
        if (mod)
        {
            allOk = allOk && mod->term();
        }
    }
    return allOk;
}
