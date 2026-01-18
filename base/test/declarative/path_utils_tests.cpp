// ============================================================
// File: test/declarative/path_utils_tests.cpp
// Unit tests for PathUtils - path validation and normalization
// ============================================================

#include <boost/test/unit_test.hpp>
#include "declarative/PathUtils.h"
#include "declarative/Metadata.h"
#include <fstream>
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;
using namespace apra;
using namespace apra::path_utils;

BOOST_AUTO_TEST_SUITE(PathUtilsTests)

// ============================================================
// Path Normalization Tests
// ============================================================

BOOST_AUTO_TEST_CASE(NormalizePath_EmptyPath_ReturnsEmpty) {
    BOOST_CHECK_EQUAL(normalizePath(""), "");
}

BOOST_AUTO_TEST_CASE(NormalizePath_SimplePath_ReturnsNormalized) {
    std::string result = normalizePath("./data/test.txt");
    // Result should be platform-appropriate
    BOOST_CHECK(!result.empty());
}

BOOST_AUTO_TEST_CASE(ParentPath_FilePath_ReturnsDirectory) {
    std::string result = parentPath("/path/to/file.txt");
    BOOST_CHECK_EQUAL(result, "/path/to");
}

BOOST_AUTO_TEST_CASE(ParentPath_EmptyPath_ReturnsEmpty) {
    BOOST_CHECK_EQUAL(parentPath(""), "");
}

BOOST_AUTO_TEST_CASE(Filename_FilePath_ReturnsFilename) {
    std::string result = filename("/path/to/file.txt");
    BOOST_CHECK_EQUAL(result, "file.txt");
}

BOOST_AUTO_TEST_CASE(Filename_EmptyPath_ReturnsEmpty) {
    BOOST_CHECK_EQUAL(filename(""), "");
}

// ============================================================
// Path Existence Tests
// ============================================================

BOOST_AUTO_TEST_CASE(PathExists_EmptyPath_ReturnsFalse) {
    BOOST_CHECK_EQUAL(pathExists(""), false);
}

BOOST_AUTO_TEST_CASE(PathExists_NonexistentPath_ReturnsFalse) {
    BOOST_CHECK_EQUAL(pathExists("/nonexistent/path/that/does/not/exist"), false);
}

BOOST_AUTO_TEST_CASE(PathExists_CurrentDirectory_ReturnsTrue) {
    BOOST_CHECK_EQUAL(pathExists("."), true);
}

BOOST_AUTO_TEST_CASE(IsFile_Directory_ReturnsFalse) {
    BOOST_CHECK_EQUAL(isFile("."), false);
}

BOOST_AUTO_TEST_CASE(IsFile_EmptyPath_ReturnsFalse) {
    BOOST_CHECK_EQUAL(isFile(""), false);
}

BOOST_AUTO_TEST_CASE(IsDirectory_CurrentDir_ReturnsTrue) {
    BOOST_CHECK_EQUAL(isDirectory("."), true);
}

BOOST_AUTO_TEST_CASE(IsDirectory_EmptyPath_ReturnsFalse) {
    BOOST_CHECK_EQUAL(isDirectory(""), false);
}

BOOST_AUTO_TEST_CASE(IsWritable_CurrentDir_ReturnsTrue) {
    // Current directory should typically be writable
    BOOST_CHECK_EQUAL(isWritable("."), true);
}

BOOST_AUTO_TEST_CASE(IsWritable_EmptyPath_ReturnsFalse) {
    BOOST_CHECK_EQUAL(isWritable(""), false);
}

// ============================================================
// Pattern Matching Tests
// ============================================================

BOOST_AUTO_TEST_CASE(HasWildcards_NoWildcards_ReturnsFalse) {
    BOOST_CHECK_EQUAL(hasWildcards("/path/to/file.txt"), false);
}

BOOST_AUTO_TEST_CASE(HasWildcards_QuestionMark_ReturnsTrue) {
    BOOST_CHECK_EQUAL(hasWildcards("/path/frame_????.jpg"), true);
}

BOOST_AUTO_TEST_CASE(HasWildcards_Asterisk_ReturnsTrue) {
    BOOST_CHECK_EQUAL(hasWildcards("/path/*.jpg"), true);
}

BOOST_AUTO_TEST_CASE(PatternDirectory_NoWildcard_ReturnsParent) {
    std::string result = patternDirectory("/path/to/file.txt");
    BOOST_CHECK_EQUAL(result, "/path/to");
}

BOOST_AUTO_TEST_CASE(PatternDirectory_WithWildcard_ReturnsDirBeforeWildcard) {
    std::string result = patternDirectory("/path/to/frame_????.jpg");
    BOOST_CHECK_EQUAL(result, "/path/to");
}

BOOST_AUTO_TEST_CASE(PatternDirectory_WildcardInDir_ReturnsDirBeforeWildcard) {
    std::string result = patternDirectory("/path/*/file.jpg");
    BOOST_CHECK_EQUAL(result, "/path");
}

BOOST_AUTO_TEST_CASE(PatternDirectory_WildcardAtStart_ReturnsDot) {
    std::string result = patternDirectory("????.jpg");
    BOOST_CHECK_EQUAL(result, ".");
}

BOOST_AUTO_TEST_CASE(CountPatternMatches_NonexistentDir_ReturnsZero) {
    BOOST_CHECK_EQUAL(countPatternMatches("/nonexistent/dir/????.jpg"), 0);
}

BOOST_AUTO_TEST_CASE(PatternHasMatches_NonexistentDir_ReturnsFalse) {
    BOOST_CHECK_EQUAL(patternHasMatches("/nonexistent/dir/????.jpg"), false);
}

// ============================================================
// Comprehensive Path Validation Tests
// ============================================================

BOOST_AUTO_TEST_CASE(ValidatePath_EmptyPath_ReturnsInvalid) {
    auto result = validatePath("", PathType::FilePath, PathRequirement::MustExist);
    BOOST_CHECK_EQUAL(result.valid, false);
    BOOST_CHECK(!result.error.empty());
}

BOOST_AUTO_TEST_CASE(ValidatePath_NetworkURL_AlwaysValid) {
    auto result = validatePath("rtsp://example.com/stream", PathType::NetworkURL, PathRequirement::None);
    BOOST_CHECK_EQUAL(result.valid, true);
}

BOOST_AUTO_TEST_CASE(ValidatePath_RequirementNone_AlwaysValid) {
    auto result = validatePath("/any/path", PathType::FilePath, PathRequirement::None);
    BOOST_CHECK_EQUAL(result.valid, true);
}

BOOST_AUTO_TEST_CASE(ValidatePath_MustExist_NonexistentFile_ReturnsInvalid) {
    auto result = validatePath("/nonexistent/file.txt", PathType::FilePath, PathRequirement::MustExist);
    BOOST_CHECK_EQUAL(result.valid, false);
    BOOST_CHECK(result.error.find("does not exist") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(ValidatePath_MustExist_ExistingDir_ReturnsValid) {
    auto result = validatePath(".", PathType::DirectoryPath, PathRequirement::MustExist);
    BOOST_CHECK_EQUAL(result.valid, true);
}

BOOST_AUTO_TEST_CASE(ValidatePath_MayExist_NonexistentFile_ReturnsValid) {
    auto result = validatePath("/nonexistent/file.txt", PathType::FilePath, PathRequirement::MayExist);
    BOOST_CHECK_EQUAL(result.valid, true);
}

BOOST_AUTO_TEST_CASE(ValidatePath_ParentMustExist_NonexistentParent_ReturnsInvalid) {
    auto result = validatePath("/nonexistent/parent/file.txt", PathType::FilePath, PathRequirement::ParentMustExist);
    BOOST_CHECK_EQUAL(result.valid, false);
    BOOST_CHECK(result.error.find("does not exist") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(ValidatePath_ParentMustExist_ExistingParent_ReturnsValid) {
    // Use current directory as parent
    auto result = validatePath("./newfile.txt", PathType::FilePath, PathRequirement::ParentMustExist);
    BOOST_CHECK_EQUAL(result.valid, true);
}

BOOST_AUTO_TEST_CASE(ValidatePath_FilePattern_NonexistentDir_ReturnsInvalid) {
    auto result = validatePath("/nonexistent/dir/frame_????.jpg", PathType::FilePattern, PathRequirement::MustExist);
    BOOST_CHECK_EQUAL(result.valid, false);
}

BOOST_AUTO_TEST_CASE(ValidatePath_MustNotExist_ExistingPath_ReturnsWarning) {
    // Current directory exists
    auto result = validatePath(".", PathType::DirectoryPath, PathRequirement::MustNotExist);
    BOOST_CHECK_EQUAL(result.valid, true);  // Valid but with warning
    BOOST_CHECK(!result.warning.empty());
}

// ============================================================
// Utility Function Tests
// ============================================================

BOOST_AUTO_TEST_CASE(PathTypeToString_AllTypes) {
    BOOST_CHECK_EQUAL(pathTypeToString(PathType::NotAPath), "NotAPath");
    BOOST_CHECK_EQUAL(pathTypeToString(PathType::FilePath), "FilePath");
    BOOST_CHECK_EQUAL(pathTypeToString(PathType::DirectoryPath), "DirectoryPath");
    BOOST_CHECK_EQUAL(pathTypeToString(PathType::FilePattern), "FilePattern");
    BOOST_CHECK_EQUAL(pathTypeToString(PathType::GlobPattern), "GlobPattern");
    BOOST_CHECK_EQUAL(pathTypeToString(PathType::DevicePath), "DevicePath");
    BOOST_CHECK_EQUAL(pathTypeToString(PathType::NetworkURL), "NetworkURL");
}

BOOST_AUTO_TEST_CASE(PathRequirementToString_AllRequirements) {
    BOOST_CHECK_EQUAL(pathRequirementToString(PathRequirement::None), "None");
    BOOST_CHECK_EQUAL(pathRequirementToString(PathRequirement::MustExist), "MustExist");
    BOOST_CHECK_EQUAL(pathRequirementToString(PathRequirement::MayExist), "MayExist");
    BOOST_CHECK_EQUAL(pathRequirementToString(PathRequirement::MustNotExist), "MustNotExist");
    BOOST_CHECK_EQUAL(pathRequirementToString(PathRequirement::ParentMustExist), "ParentMustExist");
    BOOST_CHECK_EQUAL(pathRequirementToString(PathRequirement::WillBeCreated), "WillBeCreated");
}

// ============================================================
// Directory Creation Tests (using temp directory)
// ============================================================

BOOST_AUTO_TEST_CASE(CreateDirectories_ExistingDir_ReturnsTrue) {
    BOOST_CHECK_EQUAL(createDirectories("."), true);
}

BOOST_AUTO_TEST_CASE(CreateDirectories_EmptyPath_ReturnsFalse) {
    BOOST_CHECK_EQUAL(createDirectories(""), false);
}

BOOST_AUTO_TEST_CASE(ValidatePath_WillBeCreated_CreatesDirectory) {
    // Create a unique temp directory path
    std::string tempDir = "./test_temp_dir_" + std::to_string(std::time(nullptr));
    std::string filePath = tempDir + "/subdir/file.txt";

    // Ensure it doesn't exist
    fs::remove_all(tempDir);
    BOOST_CHECK_EQUAL(isDirectory(tempDir), false);

    // Validate with WillBeCreated - should create parent directories
    auto result = validatePath(filePath, PathType::FilePath, PathRequirement::WillBeCreated);
    BOOST_CHECK_EQUAL(result.valid, true);

    // Parent directory should now exist
    std::string parentDir = parentPath(filePath);
    BOOST_CHECK_EQUAL(isDirectory(parentDir), true);
    BOOST_CHECK_EQUAL(result.directory_created, true);

    // Cleanup
    fs::remove_all(tempDir);
}

BOOST_AUTO_TEST_SUITE_END()
