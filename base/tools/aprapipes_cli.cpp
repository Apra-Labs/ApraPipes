// ============================================================
// aprapipes CLI - Command-line interface for declarative pipelines
// Task E1: CLI Tool
// ============================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <atomic>
#include <thread>
#include <chrono>
#include <csignal>
#include <cstdlib>

#include "declarative/TomlParser.h"
#include "declarative/ModuleFactory.h"
#include "declarative/ModuleRegistry.h"
#include "declarative/ModuleRegistrations.h"
#include "declarative/PipelineDescription.h"

using namespace apra;

// ============================================================
// Global state for signal handling
// ============================================================
static std::atomic<bool> g_running{true};
static std::atomic<bool> g_interrupted{false};

void signalHandler(int signum) {
    if (signum == SIGINT || signum == SIGTERM) {
        if (g_interrupted.exchange(true)) {
            // Second signal - force exit
            std::cerr << "\nForce exit.\n";
            std::exit(128 + signum);
        }
        std::cerr << "\nShutting down... (press Ctrl+C again to force)\n";
        g_running = false;
    }
}

// ============================================================
// Exit codes
// ============================================================
enum ExitCode {
    EXIT_SUCCESS_CODE = 0,
    EXIT_VALIDATION_ERROR = 1,
    EXIT_RUNTIME_ERROR = 2,
    EXIT_USAGE_ERROR = 3
};

// ============================================================
// Helper functions
// ============================================================

std::string join(const std::vector<std::string>& vec, const std::string& delim) {
    if (vec.empty()) return "";
    std::ostringstream oss;
    oss << vec[0];
    for (size_t i = 1; i < vec.size(); ++i) {
        oss << delim << vec[i];
    }
    return oss.str();
}

std::string categoryToString(ModuleCategory cat) {
    switch (cat) {
        case ModuleCategory::Source: return "source";
        case ModuleCategory::Sink: return "sink";
        case ModuleCategory::Transform: return "transform";
        case ModuleCategory::Analytics: return "analytics";
        case ModuleCategory::Controller: return "controller";
        case ModuleCategory::Utility: return "utility";
    }
    return "unknown";
}

ModuleCategory parseCategory(const std::string& str) {
    if (str == "source") return ModuleCategory::Source;
    if (str == "sink") return ModuleCategory::Sink;
    if (str == "transform") return ModuleCategory::Transform;
    if (str == "analytics") return ModuleCategory::Analytics;
    if (str == "controller") return ModuleCategory::Controller;
    if (str == "utility") return ModuleCategory::Utility;
    return ModuleCategory::Utility;  // Default
}

bool fileExists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

// Parse --set arguments into a map
// Format: "module.property=value"
bool parseOverride(const std::string& arg, std::string& module,
                   std::string& property, std::string& value) {
    auto eqPos = arg.find('=');
    if (eqPos == std::string::npos) return false;

    std::string path = arg.substr(0, eqPos);
    value = arg.substr(eqPos + 1);

    auto dotPos = path.find('.');
    if (dotPos == std::string::npos) return false;

    module = path.substr(0, dotPos);
    property = path.substr(dotPos + 1);

    return !module.empty() && !property.empty();
}

// Apply property override to description
void applyOverride(PipelineDescription& desc,
                   const std::string& module,
                   const std::string& property,
                   const std::string& value) {
    auto* inst = desc.findModule(module);
    if (!inst) {
        std::cerr << "Warning: Module '" << module << "' not found for override\n";
        return;
    }

    // Try to parse value as different types
    // First try int, then double, then bool, then string
    try {
        // Check for bool
        if (value == "true" || value == "True" || value == "TRUE") {
            inst->properties[property] = true;
            return;
        }
        if (value == "false" || value == "False" || value == "FALSE") {
            inst->properties[property] = false;
            return;
        }

        // Check for integer (no decimal point)
        if (value.find('.') == std::string::npos) {
            try {
                int64_t intVal = std::stoll(value);
                inst->properties[property] = intVal;
                return;
            } catch (...) {}
        }

        // Try double
        try {
            double dblVal = std::stod(value);
            inst->properties[property] = dblVal;
            return;
        } catch (...) {}

        // Default to string
        inst->properties[property] = value;
    } catch (...) {
        inst->properties[property] = value;
    }
}

// ============================================================
// Command: validate
// ============================================================
int cmdValidate(const std::string& filepath, bool jsonOutput) {
    if (!fileExists(filepath)) {
        if (jsonOutput) {
            std::cout << R"({"success":false,"error":"File not found: )" << filepath << R"("})" << "\n";
        } else {
            std::cerr << "Error: File not found: " << filepath << "\n";
        }
        return EXIT_VALIDATION_ERROR;
    }

    TomlParser parser;
    auto result = parser.parseFile(filepath);

    if (!result.success) {
        if (jsonOutput) {
            std::cout << R"({"success":false,"error":")" << result.error
                      << R"(","line":)" << result.error_line << "}\n";
        } else {
            std::cerr << "Parse error";
            if (result.error_line > 0) {
                std::cerr << " at line " << result.error_line;
                if (result.error_column > 0) {
                    std::cerr << ", column " << result.error_column;
                }
            }
            std::cerr << ": " << result.error << "\n";
        }
        return EXIT_VALIDATION_ERROR;
    }

    // Optionally validate with factory (builds but doesn't run)
    ModuleFactory factory;
    auto buildResult = factory.build(result.description);

    bool hasErrors = buildResult.hasErrors();
    bool hasWarnings = buildResult.hasWarnings();

    if (jsonOutput) {
        std::cout << R"({"success":)" << (hasErrors ? "false" : "true")
                  << R"(,"modules":)" << result.description.modules.size()
                  << R"(,"connections":)" << result.description.connections.size()
                  << R"(,"issues":[)";

        bool first = true;
        for (const auto& issue : buildResult.issues) {
            if (!first) std::cout << ",";
            first = false;
            std::cout << R"({"level":")"
                      << (issue.level == BuildIssue::Level::Error ? "error" :
                          issue.level == BuildIssue::Level::Warning ? "warning" : "info")
                      << R"(","code":")" << issue.code
                      << R"(","location":")" << issue.location
                      << R"(","message":")" << issue.message << "\"}";
        }
        std::cout << "]}\n";
    } else {
        // Print issues
        for (const auto& issue : buildResult.issues) {
            const char* prefix =
                issue.level == BuildIssue::Level::Error ? "[ERROR]" :
                issue.level == BuildIssue::Level::Warning ? "[WARN] " : "[INFO] ";
            std::cerr << prefix << " " << issue.code << " @ " << issue.location
                      << ": " << issue.message << "\n";
        }

        if (hasErrors) {
            std::cerr << "\n✗ Validation failed with errors\n";
        } else if (hasWarnings) {
            std::cout << "\n⚠ Validation passed with warnings\n";
            std::cout << "  Modules: " << result.description.modules.size() << "\n";
            std::cout << "  Connections: " << result.description.connections.size() << "\n";
        } else {
            std::cout << "✓ Validation successful\n";
            std::cout << "  Modules: " << result.description.modules.size() << "\n";
            std::cout << "  Connections: " << result.description.connections.size() << "\n";
        }
    }

    return hasErrors ? EXIT_VALIDATION_ERROR : EXIT_SUCCESS_CODE;
}

// ============================================================
// Command: run
// ============================================================
int cmdRun(const std::string& filepath,
           const std::vector<std::pair<std::string, std::string>>& overrides,
           bool verbose) {
    if (!fileExists(filepath)) {
        std::cerr << "Error: File not found: " << filepath << "\n";
        return EXIT_VALIDATION_ERROR;
    }

    // Parse
    TomlParser parser;
    auto parseResult = parser.parseFile(filepath);

    if (!parseResult.success) {
        std::cerr << "Parse error: " << parseResult.error << "\n";
        return EXIT_VALIDATION_ERROR;
    }

    // Apply overrides
    for (const auto& [path, value] : overrides) {
        std::string module, property, val;
        if (parseOverride(path + "=" + value, module, property, val)) {
            applyOverride(parseResult.description, module, property, val);
            if (verbose) {
                std::cout << "Override: " << module << "." << property << " = " << value << "\n";
            }
        }
    }

    // Build
    ModuleFactory::Options opts;
    opts.collect_info_messages = verbose;
    ModuleFactory factory(opts);
    auto buildResult = factory.build(parseResult.description);

    // Print issues
    for (const auto& issue : buildResult.issues) {
        if (issue.level == BuildIssue::Level::Info && !verbose) continue;
        const char* prefix =
            issue.level == BuildIssue::Level::Error ? "[ERROR]" :
            issue.level == BuildIssue::Level::Warning ? "[WARN] " : "[INFO] ";
        std::cerr << prefix << " " << issue.message << "\n";
    }

    if (!buildResult.success()) {
        std::cerr << "Failed to build pipeline\n";
        return EXIT_VALIDATION_ERROR;
    }

    // Setup signal handlers
    std::signal(SIGINT, signalHandler);
#ifndef _WIN32
    std::signal(SIGTERM, signalHandler);
#endif

    // Run
    try {
        if (!buildResult.pipeline->init()) {
            std::cerr << "Failed to initialize pipeline\n";
            return EXIT_RUNTIME_ERROR;
        }

        buildResult.pipeline->run_all_threaded();

        std::cout << "Pipeline running. Press Ctrl+C to stop.\n";

        while (g_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        std::cout << "Stopping pipeline...\n";
        buildResult.pipeline->stop();
        buildResult.pipeline->term();
        buildResult.pipeline->wait_for_all();

        std::cout << "Pipeline stopped.\n";
        return EXIT_SUCCESS_CODE;

    } catch (const std::exception& e) {
        std::cerr << "Runtime error: " << e.what() << "\n";
        return EXIT_RUNTIME_ERROR;
    }
}

// ============================================================
// Command: list-modules
// ============================================================
int cmdListModules(const std::string& category, const std::string& tag, bool jsonOutput) {
    ensureBuiltinModulesRegistered();
    auto& registry = ModuleRegistry::instance();

    std::vector<std::string> modules;

    if (!tag.empty()) {
        modules = registry.getModulesByTag(tag);
    } else if (!category.empty()) {
        modules = registry.getModulesByCategory(parseCategory(category));
    } else {
        modules = registry.getAllModules();
    }

    if (jsonOutput) {
        std::cout << R"({"modules":[)";
        bool first = true;
        for (const auto& name : modules) {
            if (!first) std::cout << ",";
            first = false;
            auto* info = registry.getModule(name);
            std::cout << R"({"name":")" << name << "\"";
            if (info) {
                std::cout << R"(,"category":")" << categoryToString(info->category) << "\"";
                std::cout << R"(,"version":")" << info->version << "\"";
            }
            std::cout << "}";
        }
        std::cout << "]}\n";
    } else {
        if (modules.empty()) {
            std::cout << "No modules found";
            if (!tag.empty()) std::cout << " with tag '" << tag << "'";
            if (!category.empty()) std::cout << " in category '" << category << "'";
            std::cout << "\n";

            // Hint about module registration
            if (registry.size() == 0) {
                std::cout << "\nNote: No modules are registered. "
                          << "Modules register themselves via REGISTER_MODULE macro.\n";
            }
        } else {
            for (const auto& name : modules) {
                auto* info = registry.getModule(name);
                std::cout << name;
                if (info) {
                    std::cout << " [" << categoryToString(info->category) << "]";
                    if (!info->tags.empty()) {
                        std::cout << " {" << join(info->tags, ", ") << "}";
                    }
                }
                std::cout << "\n";
            }
        }
    }

    return EXIT_SUCCESS_CODE;
}

// ============================================================
// Command: describe
// ============================================================
int cmdDescribe(const std::string& moduleName, bool jsonOutput) {
    ensureBuiltinModulesRegistered();
    auto& registry = ModuleRegistry::instance();
    auto* info = registry.getModule(moduleName);

    if (!info) {
        if (jsonOutput) {
            std::cout << R"({"error":"Unknown module: )" << moduleName << "\"}\n";
        } else {
            std::cerr << "Unknown module: " << moduleName << "\n";

            // Suggest similar modules
            auto allModules = registry.getAllModules();
            if (!allModules.empty()) {
                std::cerr << "\nAvailable modules:\n";
                for (const auto& m : allModules) {
                    std::cerr << "  " << m << "\n";
                }
            }
        }
        return EXIT_VALIDATION_ERROR;
    }

    if (jsonOutput) {
        // JSON output
        std::cout << "{";
        std::cout << R"("name":")" << info->name << "\"";
        std::cout << R"(,"category":")" << categoryToString(info->category) << "\"";
        std::cout << R"(,"version":")" << info->version << "\"";
        std::cout << R"(,"description":")" << info->description << "\"";

        std::cout << R"(,"tags":[)";
        bool first = true;
        for (const auto& tag : info->tags) {
            if (!first) std::cout << ",";
            first = false;
            std::cout << "\"" << tag << "\"";
        }
        std::cout << "]";

        std::cout << R"(,"inputs":[)";
        first = true;
        for (const auto& pin : info->inputs) {
            if (!first) std::cout << ",";
            first = false;
            std::cout << R"({"name":")" << pin.name << "\"";
            std::cout << R"(,"required":)" << (pin.required ? "true" : "false");
            std::cout << R"(,"frameTypes":[)";
            bool firstType = true;
            for (const auto& ft : pin.frame_types) {
                if (!firstType) std::cout << ",";
                firstType = false;
                std::cout << "\"" << ft << "\"";
            }
            std::cout << "]}";
        }
        std::cout << "]";

        std::cout << R"(,"outputs":[)";
        first = true;
        for (const auto& pin : info->outputs) {
            if (!first) std::cout << ",";
            first = false;
            std::cout << R"({"name":")" << pin.name << "\"";
            std::cout << R"(,"frameTypes":[)";
            bool firstType = true;
            for (const auto& ft : pin.frame_types) {
                if (!firstType) std::cout << ",";
                firstType = false;
                std::cout << "\"" << ft << "\"";
            }
            std::cout << "]}";
        }
        std::cout << "]";

        std::cout << R"(,"properties":[)";
        first = true;
        for (const auto& prop : info->properties) {
            if (!first) std::cout << ",";
            first = false;
            std::cout << R"({"name":")" << prop.name << "\"";
            std::cout << R"(,"type":")" << prop.type << "\"";
            std::cout << R"(,"mutability":")" << prop.mutability << "\"";
            std::cout << R"(,"default":")" << prop.default_value << "\"";
            if (!prop.min_value.empty()) {
                std::cout << R"(,"min":")" << prop.min_value << "\"";
                std::cout << R"(,"max":")" << prop.max_value << "\"";
            }
            if (!prop.enum_values.empty()) {
                std::cout << R"(,"enumValues":[)";
                bool firstEnum = true;
                for (const auto& ev : prop.enum_values) {
                    if (!firstEnum) std::cout << ",";
                    firstEnum = false;
                    std::cout << "\"" << ev << "\"";
                }
                std::cout << "]";
            }
            std::cout << "}";
        }
        std::cout << "]";

        std::cout << "}\n";
    } else {
        // Human-readable output
        std::cout << "# " << info->name << "\n\n";
        std::cout << "Category: " << categoryToString(info->category) << "\n";
        std::cout << "Version:  " << info->version << "\n";
        if (!info->tags.empty()) {
            std::cout << "Tags:     " << join(info->tags, ", ") << "\n";
        }
        std::cout << "\n" << info->description << "\n";

        if (!info->inputs.empty()) {
            std::cout << "\n## Inputs\n";
            for (const auto& pin : info->inputs) {
                std::cout << "  - " << pin.name;
                if (!pin.frame_types.empty()) {
                    std::cout << " [" << join(pin.frame_types, ", ") << "]";
                }
                if (pin.required) std::cout << " (required)";
                if (!pin.description.empty()) {
                    std::cout << "\n    " << pin.description;
                }
                std::cout << "\n";
            }
        }

        if (!info->outputs.empty()) {
            std::cout << "\n## Outputs\n";
            for (const auto& pin : info->outputs) {
                std::cout << "  - " << pin.name;
                if (!pin.frame_types.empty()) {
                    std::cout << " [" << join(pin.frame_types, ", ") << "]";
                }
                if (!pin.description.empty()) {
                    std::cout << "\n    " << pin.description;
                }
                std::cout << "\n";
            }
        }

        if (!info->properties.empty()) {
            std::cout << "\n## Properties\n";
            for (const auto& prop : info->properties) {
                std::cout << "  - " << prop.name << " (" << prop.type << ")";
                std::cout << " [" << prop.mutability << "]";
                std::cout << "\n    Default: " << prop.default_value;
                if (!prop.min_value.empty()) {
                    std::cout << "  Range: [" << prop.min_value << ", " << prop.max_value << "]";
                }
                if (!prop.enum_values.empty()) {
                    std::cout << "  Values: " << join(prop.enum_values, " | ");
                }
                if (!prop.unit.empty()) {
                    std::cout << "  Unit: " << prop.unit;
                }
                if (!prop.description.empty()) {
                    std::cout << "\n    " << prop.description;
                }
                std::cout << "\n";
            }
        }
    }

    return EXIT_SUCCESS_CODE;
}

// ============================================================
// Help text
// ============================================================
void printUsage(const char* progname) {
    std::cout << R"(
ApraPipes - Declarative Pipeline CLI

Usage: )" << progname << R"( <command> [options]

Commands:
  validate <file.toml>          Validate a pipeline definition file
  run <file.toml>               Build and run a pipeline
  list-modules                  List all registered modules
  describe <module>             Show detailed module information

Options:
  --help, -h                    Show this help message
  --json                        Output in JSON format (for tooling)
  --verbose, -v                 Show detailed output

Options for 'run':
  --set <module.prop>=<value>   Override a property value at runtime

Options for 'list-modules':
  --category <category>         Filter by category (source, sink, transform,
                                analytics, controller, utility)
  --tag <tag>                   Filter by tag

Examples:
  )" << progname << R"( validate pipeline.toml
  )" << progname << R"( run pipeline.toml --set decoder.device_id=1
  )" << progname << R"( list-modules --category source
  )" << progname << R"( describe FileReaderModule --json

Exit Codes:
  0 - Success
  1 - Validation error
  2 - Runtime error
  3 - Usage error
)";
}

// ============================================================
// Main entry point
// ============================================================
int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return EXIT_USAGE_ERROR;
    }

    std::string command = argv[1];

    // Check for help
    if (command == "--help" || command == "-h" || command == "help") {
        printUsage(argv[0]);
        return EXIT_SUCCESS_CODE;
    }

    // Parse global options
    bool jsonOutput = false;
    bool verbose = false;

    // Collect remaining args after command
    std::vector<std::string> args;
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--json") {
            jsonOutput = true;
        } else if (arg == "--verbose" || arg == "-v") {
            verbose = true;
        } else {
            args.push_back(arg);
        }
    }

    // Dispatch commands
    if (command == "validate") {
        if (args.empty()) {
            std::cerr << "Error: validate requires a file path\n";
            return EXIT_USAGE_ERROR;
        }
        return cmdValidate(args[0], jsonOutput);
    }
    else if (command == "run") {
        if (args.empty()) {
            std::cerr << "Error: run requires a file path\n";
            return EXIT_USAGE_ERROR;
        }

        // Parse --set arguments
        std::vector<std::pair<std::string, std::string>> overrides;
        std::string filepath = args[0];

        for (size_t i = 1; i < args.size(); ++i) {
            if (args[i] == "--set" && i + 1 < args.size()) {
                std::string override = args[++i];
                auto eqPos = override.find('=');
                if (eqPos != std::string::npos) {
                    overrides.emplace_back(
                        override.substr(0, eqPos),
                        override.substr(eqPos + 1)
                    );
                }
            }
        }

        return cmdRun(filepath, overrides, verbose);
    }
    else if (command == "list-modules") {
        std::string category, tag;

        for (size_t i = 0; i < args.size(); ++i) {
            if (args[i] == "--category" && i + 1 < args.size()) {
                category = args[++i];
            } else if (args[i] == "--tag" && i + 1 < args.size()) {
                tag = args[++i];
            }
        }

        return cmdListModules(category, tag, jsonOutput);
    }
    else if (command == "describe") {
        if (args.empty()) {
            std::cerr << "Error: describe requires a module name\n";
            return EXIT_USAGE_ERROR;
        }
        return cmdDescribe(args[0], jsonOutput);
    }
    else {
        std::cerr << "Unknown command: " << command << "\n";
        printUsage(argv[0]);
        return EXIT_USAGE_ERROR;
    }
}
