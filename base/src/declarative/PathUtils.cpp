// ============================================================
// File: declarative/PathUtils.cpp
// Path validation and normalization utilities implementation
// ============================================================

#include "declarative/PathUtils.h"
#include <boost/filesystem.hpp>
#include <algorithm>
#include <fstream>

#ifdef _WIN32
#include <io.h>
#define access _access
#define W_OK 2
#else
#include <unistd.h>
#endif

namespace apra {
namespace path_utils {

namespace fs = boost::filesystem;

// ============================================================
// Path Normalization
// ============================================================

std::string normalizePath(const std::string& path) {
    if (path.empty()) return path;

    // Use boost::filesystem::path which handles cross-platform normalization
    fs::path p(path);
    return p.make_preferred().string();
}

std::string parentPath(const std::string& path) {
    if (path.empty()) return "";
    fs::path p(path);
    return p.parent_path().string();
}

std::string filename(const std::string& path) {
    if (path.empty()) return "";
    fs::path p(path);
    return p.filename().string();
}

// ============================================================
// Path Existence Checks
// ============================================================

bool pathExists(const std::string& path) {
    if (path.empty()) return false;
    try {
        return fs::exists(path);
    } catch (...) {
        return false;
    }
}

bool isFile(const std::string& path) {
    if (path.empty()) return false;
    try {
        return fs::is_regular_file(path);
    } catch (...) {
        return false;
    }
}

bool isDirectory(const std::string& path) {
    if (path.empty()) return false;
    try {
        return fs::is_directory(path);
    } catch (...) {
        return false;
    }
}

bool isWritable(const std::string& path) {
    if (path.empty()) return false;

    std::string pathToCheck = path;

    // If the path doesn't exist, check if parent is writable
    if (!pathExists(path)) {
        std::string parent = parentPath(path);
        if (parent.empty() || parent == path) {
            // Root or current directory
            parent = ".";
        }
        if (!pathExists(parent)) {
            return false;
        }
        pathToCheck = parent;
    }

    // Check write permission
    return access(pathToCheck.c_str(), W_OK) == 0;
}

// ============================================================
// Directory Operations
// ============================================================

bool createDirectories(const std::string& path) {
    if (path.empty()) return false;
    try {
        if (fs::exists(path)) {
            return fs::is_directory(path);
        }
        return fs::create_directories(path);
    } catch (...) {
        return false;
    }
}

// ============================================================
// Pattern Matching
// ============================================================

bool hasWildcards(const std::string& path) {
    return path.find('?') != std::string::npos ||
           path.find('*') != std::string::npos;
}

std::string patternDirectory(const std::string& pattern) {
    // Find the last separator before any wildcard
    size_t wildcardPos = pattern.find_first_of("?*");
    if (wildcardPos == std::string::npos) {
        // No wildcards, return parent directory
        return parentPath(pattern);
    }

    // Find the last separator before the wildcard
    size_t sepPos = pattern.find_last_of("/\\", wildcardPos);
    if (sepPos == std::string::npos) {
        return ".";
    }
    return pattern.substr(0, sepPos);
}

// Helper: Expand a pattern like "frame_????.jpg" to a regex-like check
// This is a simple implementation that handles ???? patterns
static bool matchesPattern(const std::string& filename, const std::string& patternFilename) {
    if (filename.length() != patternFilename.length()) {
        return false;
    }

    for (size_t i = 0; i < filename.length(); ++i) {
        if (patternFilename[i] == '?') {
            // ? matches any single character (but we expect digits)
            if (!std::isdigit(filename[i])) {
                return false;
            }
        } else if (patternFilename[i] != filename[i]) {
            return false;
        }
    }
    return true;
}

bool patternHasMatches(const std::string& pattern) {
    return countPatternMatches(pattern) > 0;
}

size_t countPatternMatches(const std::string& pattern) {
    if (!hasWildcards(pattern)) {
        // Not a pattern, check if file exists
        return pathExists(pattern) ? 1 : 0;
    }

    std::string dir = patternDirectory(pattern);
    std::string patternFilename = filename(pattern);

    if (!isDirectory(dir)) {
        return 0;
    }

    size_t count = 0;
    try {
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (fs::is_regular_file(entry.path())) {
                std::string fname = entry.path().filename().string();
                if (matchesPattern(fname, patternFilename)) {
                    count++;
                }
            }
        }
    } catch (...) {
        return 0;
    }

    return count;
}

std::string firstPatternMatch(const std::string& pattern) {
    if (!hasWildcards(pattern)) {
        return pathExists(pattern) ? pattern : "";
    }

    std::string dir = patternDirectory(pattern);
    std::string patternFilename = filename(pattern);

    if (!isDirectory(dir)) {
        return "";
    }

    try {
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (fs::is_regular_file(entry.path())) {
                std::string fname = entry.path().filename().string();
                if (matchesPattern(fname, patternFilename)) {
                    return entry.path().string();
                }
            }
        }
    } catch (...) {
        return "";
    }

    return "";
}

// ============================================================
// Comprehensive Path Validation
// ============================================================

PathValidationResult validatePath(
    const std::string& path,
    PathType type,
    PathRequirement requirement
) {
    PathValidationResult result;
    result.normalized_path = normalizePath(path);

    // Empty path check
    if (path.empty()) {
        result.valid = false;
        result.error = "Path is empty";
        return result;
    }

    // Network URLs don't need filesystem validation
    if (type == PathType::NetworkURL) {
        result.valid = true;
        return result;
    }

    // No validation needed
    if (requirement == PathRequirement::None) {
        result.valid = true;
        return result;
    }

    // Get parent directory for patterns and files
    std::string parentDir;
    if (type == PathType::DirectoryPath) {
        parentDir = parentPath(path);
        if (parentDir.empty()) parentDir = ".";
    } else {
        parentDir = patternDirectory(path);
        if (parentDir.empty()) parentDir = ".";
    }

    switch (requirement) {
        case PathRequirement::MustExist: {
            if (type == PathType::FilePattern || type == PathType::GlobPattern) {
                // For patterns, check if any files match
                if (!isDirectory(parentDir)) {
                    result.valid = false;
                    result.error = "Directory does not exist: " + parentDir;
                } else if (!patternHasMatches(path)) {
                    // Warning, not error - per user feedback
                    result.valid = true;
                    result.warning = "No files match pattern: " + path;
                } else {
                    result.valid = true;
                }
            } else if (type == PathType::DirectoryPath) {
                if (!isDirectory(path)) {
                    result.valid = false;
                    result.error = "Directory does not exist: " + path;
                } else {
                    result.valid = true;
                }
            } else {
                // FilePath, DevicePath
                if (!pathExists(path)) {
                    result.valid = false;
                    result.error = "File does not exist: " + path;
                } else if (type == PathType::FilePath && !isFile(path)) {
                    result.valid = false;
                    result.error = "Path is not a file: " + path;
                } else {
                    result.valid = true;
                }
            }
            break;
        }

        case PathRequirement::MayExist: {
            // No existence validation needed
            result.valid = true;
            break;
        }

        case PathRequirement::MustNotExist: {
            if (type == PathType::FilePattern || type == PathType::GlobPattern) {
                if (patternHasMatches(path)) {
                    // Warning, not error
                    result.valid = true;
                    result.warning = "Files already match pattern (will be overwritten): " + path;
                } else {
                    result.valid = true;
                }
            } else {
                if (pathExists(path)) {
                    // Warning, not error
                    result.valid = true;
                    result.warning = "Path already exists (will be overwritten): " + path;
                } else {
                    result.valid = true;
                }
            }
            break;
        }

        case PathRequirement::ParentMustExist: {
            if (!isDirectory(parentDir)) {
                result.valid = false;
                result.error = "Parent directory does not exist: " + parentDir;
            } else if (!isWritable(parentDir)) {
                result.valid = false;
                result.error = "Parent directory is not writable: " + parentDir;
            } else {
                result.valid = true;
            }
            break;
        }

        case PathRequirement::WillBeCreated: {
            // Try to create the parent directory
            if (!isDirectory(parentDir)) {
                if (createDirectories(parentDir)) {
                    result.directory_created = true;
                    result.valid = true;
                } else {
                    result.valid = false;
                    result.error = "Failed to create directory: " + parentDir;
                }
            } else {
                result.valid = true;
            }

            // Check write permissions
            if (result.valid && !isWritable(parentDir)) {
                result.valid = false;
                result.error = "Directory is not writable: " + parentDir;
            }
            break;
        }

        default:
            result.valid = true;
            break;
    }

    return result;
}

// ============================================================
// Utility Functions
// ============================================================

std::string pathTypeToString(PathType type) {
    switch (type) {
        case PathType::NotAPath: return "NotAPath";
        case PathType::FilePath: return "FilePath";
        case PathType::DirectoryPath: return "DirectoryPath";
        case PathType::FilePattern: return "FilePattern";
        case PathType::GlobPattern: return "GlobPattern";
        case PathType::DevicePath: return "DevicePath";
        case PathType::NetworkURL: return "NetworkURL";
    }
    return "Unknown";
}

std::string pathRequirementToString(PathRequirement requirement) {
    switch (requirement) {
        case PathRequirement::None: return "None";
        case PathRequirement::MustExist: return "MustExist";
        case PathRequirement::MayExist: return "MayExist";
        case PathRequirement::MustNotExist: return "MustNotExist";
        case PathRequirement::ParentMustExist: return "ParentMustExist";
        case PathRequirement::WillBeCreated: return "WillBeCreated";
    }
    return "Unknown";
}

} // namespace path_utils
} // namespace apra
