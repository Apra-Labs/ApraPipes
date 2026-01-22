// ============================================================
// File: declarative/PathUtils.h
// Path validation and normalization utilities for declarative pipelines
// ============================================================

#pragma once

#include <string>
#include <vector>
#include "Metadata.h"

namespace apra {
namespace path_utils {

// ============================================================
// Path Validation Result
// ============================================================
struct PathValidationResult {
    bool valid = false;
    std::string error;           // Error message if not valid
    std::string warning;         // Warning message (e.g., no files match pattern)
    std::string normalized_path; // Platform-normalized path
    bool directory_created = false; // True if directory was created
};

// ============================================================
// Path Normalization
// ============================================================

// Normalize path separators to platform-native format
// On Windows: converts / to \\
// On Linux/macOS: converts \\ to /
std::string normalizePath(const std::string& path);

// Get the parent directory of a path
// e.g., "/path/to/file.txt" -> "/path/to"
std::string parentPath(const std::string& path);

// Get the filename component of a path
// e.g., "/path/to/file.txt" -> "file.txt"
std::string filename(const std::string& path);

// ============================================================
// Path Existence Checks
// ============================================================

// Check if a path exists (file or directory)
bool pathExists(const std::string& path);

// Check if path is a regular file
bool isFile(const std::string& path);

// Check if path is a directory
bool isDirectory(const std::string& path);

// Check if path is writable (can create/write files)
bool isWritable(const std::string& path);

// ============================================================
// Directory Operations
// ============================================================

// Create directory and all parent directories if needed
// Returns true if directory exists or was created successfully
bool createDirectories(const std::string& path);

// ============================================================
// Pattern Matching
// ============================================================

// Check if any files match a pattern with ???? wildcards
// e.g., "/path/frame_????.jpg" checks for frame_0000.jpg, frame_0001.jpg, etc.
bool patternHasMatches(const std::string& pattern);

// Count how many files match a pattern
size_t countPatternMatches(const std::string& pattern);

// Get first matching file for a pattern (for existence check)
std::string firstPatternMatch(const std::string& pattern);

// ============================================================
// Comprehensive Path Validation
// ============================================================

// Validate a path based on its type and requirement
// This is the main entry point for path validation
//
// Validation rules:
// - MustExist: Path must exist (error if not, warn for patterns with no matches)
// - MayExist: No existence check needed
// - MustNotExist: Path must not exist (warn if exists)
// - ParentMustExist: Parent directory must exist (error if not)
// - WillBeCreated: Attempt to create parent directory (error if fails)
//
// Additional checks:
// - For writers (WillBeCreated/ParentMustExist): Check write permissions
// - For patterns: Check if at least one file matches
// - Normalize path separators for cross-platform compatibility
//
PathValidationResult validatePath(
    const std::string& path,
    PathType type,
    PathRequirement requirement
);

// ============================================================
// Utility Functions
// ============================================================

// Convert PathType enum to string for error messages
std::string pathTypeToString(PathType type);

// Convert PathRequirement enum to string for error messages
std::string pathRequirementToString(PathRequirement requirement);

// Check if a path contains wildcard characters (? or *)
bool hasWildcards(const std::string& path);

// Extract the directory part from a file pattern
// e.g., "./data/testOutput/bmp_????.bmp" -> "./data/testOutput"
std::string patternDirectory(const std::string& pattern);

} // namespace path_utils
} // namespace apra
