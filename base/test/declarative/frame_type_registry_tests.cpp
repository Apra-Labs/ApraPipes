// ============================================================
// Unit tests for declarative/FrameTypeRegistry.h
// Task A3: FrameType Registry
// ============================================================

#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include "declarative/FrameTypeRegistry.h"
#include <thread>
#include <vector>

using namespace apra;

BOOST_AUTO_TEST_SUITE(FrameTypeRegistryTests)

// ============================================================
// Fixture to clear registry between tests
// ============================================================
struct RegistryFixture {
    RegistryFixture() {
        FrameTypeRegistry::instance().clear();
        registerTestTypes();
    }
    ~RegistryFixture() {
        FrameTypeRegistry::instance().clear();
    }

    void registerTestTypes() {
        // Register a hierarchy: Frame -> VideoFrame -> RawImagePlanar
        //                                           -> EncodedVideoFrame -> H264Frame
        FrameTypeInfo frame;
        frame.name = "Frame";
        frame.parent = "";
        frame.description = "Base frame type";
        frame.tags = {"base"};
        FrameTypeRegistry::instance().registerFrameType(frame);

        FrameTypeInfo videoFrame;
        videoFrame.name = "VideoFrame";
        videoFrame.parent = "Frame";
        videoFrame.description = "Video frame type";
        videoFrame.tags = {"video"};
        FrameTypeRegistry::instance().registerFrameType(videoFrame);

        FrameTypeInfo rawImage;
        rawImage.name = "RawImagePlanar";
        rawImage.parent = "VideoFrame";
        rawImage.description = "Raw planar image";
        rawImage.tags = {"video", "raw", "planar"};
        FrameTypeRegistry::instance().registerFrameType(rawImage);

        FrameTypeInfo encodedVideo;
        encodedVideo.name = "EncodedVideoFrame";
        encodedVideo.parent = "VideoFrame";
        encodedVideo.description = "Encoded video frame";
        encodedVideo.tags = {"video", "encoded"};
        FrameTypeRegistry::instance().registerFrameType(encodedVideo);

        FrameTypeInfo h264;
        h264.name = "H264Frame";
        h264.parent = "EncodedVideoFrame";
        h264.description = "H.264/AVC encoded frame";
        h264.tags = {"video", "encoded", "h264"};
        FrameTypeRegistry::instance().registerFrameType(h264);

        FrameTypeInfo audioFrame;
        audioFrame.name = "AudioFrame";
        audioFrame.parent = "Frame";
        audioFrame.description = "Audio frame type";
        audioFrame.tags = {"audio"};
        FrameTypeRegistry::instance().registerFrameType(audioFrame);
    }
};

// ============================================================
// Singleton Tests
// ============================================================

BOOST_AUTO_TEST_CASE(Instance_ReturnsSingleton)
{
    auto& reg1 = FrameTypeRegistry::instance();
    auto& reg2 = FrameTypeRegistry::instance();
    BOOST_CHECK_EQUAL(&reg1, &reg2);
}

// ============================================================
// Registration Tests
// ============================================================

BOOST_FIXTURE_TEST_CASE(RegisterFrameType_AddsToRegistry, RegistryFixture)
{
    BOOST_CHECK(FrameTypeRegistry::instance().hasFrameType("Frame"));
    BOOST_CHECK(FrameTypeRegistry::instance().hasFrameType("VideoFrame"));
    BOOST_CHECK(FrameTypeRegistry::instance().hasFrameType("H264Frame"));
}

BOOST_FIXTURE_TEST_CASE(RegisterFrameType_DuplicateIgnored, RegistryFixture)
{
    size_t countBefore = FrameTypeRegistry::instance().size();

    FrameTypeInfo duplicate;
    duplicate.name = "Frame";
    duplicate.description = "Different description";
    FrameTypeRegistry::instance().registerFrameType(duplicate);

    BOOST_CHECK_EQUAL(FrameTypeRegistry::instance().size(), countBefore);

    // Original should be preserved
    auto* info = FrameTypeRegistry::instance().getFrameType("Frame");
    BOOST_CHECK_EQUAL(info->description, "Base frame type");
}

// ============================================================
// Basic Query Tests
// ============================================================

BOOST_FIXTURE_TEST_CASE(HasFrameType_ReturnsTrue, RegistryFixture)
{
    BOOST_CHECK(FrameTypeRegistry::instance().hasFrameType("H264Frame"));
}

BOOST_FIXTURE_TEST_CASE(HasFrameType_ReturnsFalse, RegistryFixture)
{
    BOOST_CHECK(!FrameTypeRegistry::instance().hasFrameType("NonExistent"));
}

BOOST_FIXTURE_TEST_CASE(GetFrameType_ReturnsInfo, RegistryFixture)
{
    auto* info = FrameTypeRegistry::instance().getFrameType("H264Frame");
    BOOST_REQUIRE(info != nullptr);
    BOOST_CHECK_EQUAL(info->name, "H264Frame");
    BOOST_CHECK_EQUAL(info->parent, "EncodedVideoFrame");
}

BOOST_FIXTURE_TEST_CASE(GetFrameType_ReturnsNullForUnknown, RegistryFixture)
{
    auto* info = FrameTypeRegistry::instance().getFrameType("Unknown");
    BOOST_CHECK(info == nullptr);
}

BOOST_FIXTURE_TEST_CASE(GetAllFrameTypes_ReturnsAll, RegistryFixture)
{
    auto types = FrameTypeRegistry::instance().getAllFrameTypes();
    BOOST_CHECK_EQUAL(types.size(), 6);  // Frame, VideoFrame, RawImagePlanar, EncodedVideoFrame, H264Frame, AudioFrame
}

// ============================================================
// Tag Query Tests
// ============================================================

BOOST_FIXTURE_TEST_CASE(GetFrameTypesByTag_ReturnsMatching, RegistryFixture)
{
    auto videoTypes = FrameTypeRegistry::instance().getFrameTypesByTag("video");
    BOOST_CHECK_GE(videoTypes.size(), 4);  // VideoFrame, RawImagePlanar, EncodedVideoFrame, H264Frame

    // Check specific types are included
    bool hasH264 = std::find(videoTypes.begin(), videoTypes.end(), "H264Frame") != videoTypes.end();
    BOOST_CHECK(hasH264);
}

BOOST_FIXTURE_TEST_CASE(GetFrameTypesByTag_ExcludesNonMatching, RegistryFixture)
{
    auto audioTypes = FrameTypeRegistry::instance().getFrameTypesByTag("audio");

    bool hasH264 = std::find(audioTypes.begin(), audioTypes.end(), "H264Frame") != audioTypes.end();
    BOOST_CHECK(!hasH264);
}

BOOST_FIXTURE_TEST_CASE(GetFrameTypesByTag_EmptyForUnknownTag, RegistryFixture)
{
    auto types = FrameTypeRegistry::instance().getFrameTypesByTag("nonexistent");
    BOOST_CHECK(types.empty());
}

// ============================================================
// Hierarchy Query Tests
// ============================================================

BOOST_FIXTURE_TEST_CASE(GetParent_ReturnsParent, RegistryFixture)
{
    BOOST_CHECK_EQUAL(FrameTypeRegistry::instance().getParent("H264Frame"), "EncodedVideoFrame");
    BOOST_CHECK_EQUAL(FrameTypeRegistry::instance().getParent("EncodedVideoFrame"), "VideoFrame");
    BOOST_CHECK_EQUAL(FrameTypeRegistry::instance().getParent("VideoFrame"), "Frame");
}

BOOST_FIXTURE_TEST_CASE(GetParent_EmptyForRoot, RegistryFixture)
{
    BOOST_CHECK(FrameTypeRegistry::instance().getParent("Frame").empty());
}

BOOST_FIXTURE_TEST_CASE(GetParent_EmptyForUnknown, RegistryFixture)
{
    BOOST_CHECK(FrameTypeRegistry::instance().getParent("Unknown").empty());
}

BOOST_FIXTURE_TEST_CASE(GetSubtypes_ReturnsDirectChildren, RegistryFixture)
{
    auto subtypes = FrameTypeRegistry::instance().getSubtypes("VideoFrame");
    BOOST_CHECK_EQUAL(subtypes.size(), 2);  // RawImagePlanar, EncodedVideoFrame

    bool hasRaw = std::find(subtypes.begin(), subtypes.end(), "RawImagePlanar") != subtypes.end();
    bool hasEncoded = std::find(subtypes.begin(), subtypes.end(), "EncodedVideoFrame") != subtypes.end();
    BOOST_CHECK(hasRaw);
    BOOST_CHECK(hasEncoded);
}

BOOST_FIXTURE_TEST_CASE(GetSubtypes_EmptyForLeaf, RegistryFixture)
{
    auto subtypes = FrameTypeRegistry::instance().getSubtypes("H264Frame");
    BOOST_CHECK(subtypes.empty());
}

BOOST_FIXTURE_TEST_CASE(GetAncestors_ReturnsAllAncestors, RegistryFixture)
{
    auto ancestors = FrameTypeRegistry::instance().getAncestors("H264Frame");
    BOOST_CHECK_EQUAL(ancestors.size(), 3);  // EncodedVideoFrame, VideoFrame, Frame

    BOOST_CHECK_EQUAL(ancestors[0], "EncodedVideoFrame");
    BOOST_CHECK_EQUAL(ancestors[1], "VideoFrame");
    BOOST_CHECK_EQUAL(ancestors[2], "Frame");
}

BOOST_FIXTURE_TEST_CASE(GetAncestors_EmptyForRoot, RegistryFixture)
{
    auto ancestors = FrameTypeRegistry::instance().getAncestors("Frame");
    BOOST_CHECK(ancestors.empty());
}

// ============================================================
// Compatibility Tests
// ============================================================

BOOST_FIXTURE_TEST_CASE(IsSubtype_TrueForSameType, RegistryFixture)
{
    BOOST_CHECK(FrameTypeRegistry::instance().isSubtype("H264Frame", "H264Frame"));
}

BOOST_FIXTURE_TEST_CASE(IsSubtype_TrueForDirectParent, RegistryFixture)
{
    BOOST_CHECK(FrameTypeRegistry::instance().isSubtype("H264Frame", "EncodedVideoFrame"));
}

BOOST_FIXTURE_TEST_CASE(IsSubtype_TrueForTransitiveAncestor, RegistryFixture)
{
    BOOST_CHECK(FrameTypeRegistry::instance().isSubtype("H264Frame", "VideoFrame"));
    BOOST_CHECK(FrameTypeRegistry::instance().isSubtype("H264Frame", "Frame"));
}

BOOST_FIXTURE_TEST_CASE(IsSubtype_FalseForUnrelatedType, RegistryFixture)
{
    BOOST_CHECK(!FrameTypeRegistry::instance().isSubtype("H264Frame", "AudioFrame"));
    BOOST_CHECK(!FrameTypeRegistry::instance().isSubtype("H264Frame", "RawImagePlanar"));
}

BOOST_FIXTURE_TEST_CASE(IsSubtype_FalseForChildOfParent, RegistryFixture)
{
    // Parent is not a subtype of child
    BOOST_CHECK(!FrameTypeRegistry::instance().isSubtype("VideoFrame", "H264Frame"));
}

BOOST_FIXTURE_TEST_CASE(IsCompatible_OutputSubtypeOfInput, RegistryFixture)
{
    // H264Frame output is compatible with EncodedVideoFrame input
    BOOST_CHECK(FrameTypeRegistry::instance().isCompatible("H264Frame", "EncodedVideoFrame"));
    BOOST_CHECK(FrameTypeRegistry::instance().isCompatible("H264Frame", "VideoFrame"));
    BOOST_CHECK(FrameTypeRegistry::instance().isCompatible("H264Frame", "Frame"));
}

BOOST_FIXTURE_TEST_CASE(IsCompatible_ExactMatch, RegistryFixture)
{
    BOOST_CHECK(FrameTypeRegistry::instance().isCompatible("H264Frame", "H264Frame"));
}

BOOST_FIXTURE_TEST_CASE(IsCompatible_NotForParentToChild, RegistryFixture)
{
    // Generic VideoFrame is NOT compatible with specific H264Frame input
    BOOST_CHECK(!FrameTypeRegistry::instance().isCompatible("VideoFrame", "H264Frame"));
}

// ============================================================
// Export Tests
// ============================================================

BOOST_FIXTURE_TEST_CASE(ToJson_ContainsAllTypes, RegistryFixture)
{
    std::string json = FrameTypeRegistry::instance().toJson();

    BOOST_CHECK(json.find("\"frameTypes\"") != std::string::npos);
    BOOST_CHECK(json.find("\"H264Frame\"") != std::string::npos);
    BOOST_CHECK(json.find("\"VideoFrame\"") != std::string::npos);
    BOOST_CHECK(json.find("\"parent\"") != std::string::npos);
}

BOOST_FIXTURE_TEST_CASE(ToMarkdown_ContainsHierarchy, RegistryFixture)
{
    std::string md = FrameTypeRegistry::instance().toMarkdown();

    BOOST_CHECK(md.find("# Frame Type Hierarchy") != std::string::npos);
    BOOST_CHECK(md.find("**Frame**") != std::string::npos);
    BOOST_CHECK(md.find("**H264Frame**") != std::string::npos);
}

// ============================================================
// Thread Safety Tests
// ============================================================

BOOST_AUTO_TEST_CASE(ConcurrentRegistration_IsThreadSafe)
{
    FrameTypeRegistry::instance().clear();

    const int numThreads = 10;
    std::vector<std::thread> threads;

    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([i]() {
            FrameTypeInfo info;
            info.name = "Type" + std::to_string(i);
            info.parent = "";
            info.description = "Test type";
            FrameTypeRegistry::instance().registerFrameType(std::move(info));
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    BOOST_CHECK_EQUAL(FrameTypeRegistry::instance().size(), numThreads);
    FrameTypeRegistry::instance().clear();
}

BOOST_FIXTURE_TEST_CASE(ConcurrentQuery_IsThreadSafe, RegistryFixture)
{
    const int numThreads = 10;
    std::vector<std::thread> threads;
    std::atomic<int> successCount{0};

    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([&successCount]() {
            for (int j = 0; j < 100; ++j) {
                auto types = FrameTypeRegistry::instance().getAllFrameTypes();
                auto ancestors = FrameTypeRegistry::instance().getAncestors("H264Frame");
                bool compat = FrameTypeRegistry::instance().isCompatible("H264Frame", "VideoFrame");

                if (types.size() >= 6 && ancestors.size() == 3 && compat) {
                    ++successCount;
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    BOOST_CHECK_EQUAL(successCount.load(), numThreads * 100);
}

// ============================================================
// Clear and Size Tests
// ============================================================

BOOST_FIXTURE_TEST_CASE(Clear_RemovesAllTypes, RegistryFixture)
{
    BOOST_CHECK_GT(FrameTypeRegistry::instance().size(), 0);

    FrameTypeRegistry::instance().clear();

    BOOST_CHECK_EQUAL(FrameTypeRegistry::instance().size(), 0);
    BOOST_CHECK(!FrameTypeRegistry::instance().hasFrameType("Frame"));
}

BOOST_AUTO_TEST_SUITE_END()
