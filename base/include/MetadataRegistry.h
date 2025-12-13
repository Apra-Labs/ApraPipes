#pragma once

#include "FrameMetadata.h"
#include "Frame.h"
#include <functional>
#include <map>
#include <vector>
#include <optional>
#include <memory>

/**
 * @brief Metadata conversion function type
 *
 * Takes a source frame and converts it to target metadata type.
 * Returns nullptr if conversion fails.
 */
using ConverterFunc = std::function<frame_sp(const frame_sp&)>;

/**
 * @brief Represents a single conversion step in a conversion path
 */
struct ConversionStep {
    FrameMetadata::FrameType sourceType;
    FrameMetadata::FrameType targetType;
    ConverterFunc converter;
    int cost;

    ConversionStep(FrameMetadata::FrameType src, FrameMetadata::FrameType tgt,
                   ConverterFunc conv, int c)
        : sourceType(src), targetType(tgt), converter(conv), cost(c) {}
};

/**
 * @brief Represents a complete conversion path from source to target type
 */
struct ConversionPath {
    std::vector<ConversionStep> steps;
    int totalCost;

    ConversionPath() : totalCost(0) {}

    void addStep(const ConversionStep& step) {
        steps.push_back(step);
        totalCost += step.cost;
    }

    bool isEmpty() const {
        return steps.empty();
    }
};

/**
 * @brief Registry for metadata type conversions with automatic path-finding
 *
 * This is Phase 2 of the Intelligent Pipeline Framework (see intelligent_pipeline_design.md).
 *
 * Features:
 * - Register direct conversions between metadata types
 * - Find shortest conversion path (supports multi-hop: A→B→C)
 * - Convert frames along best path
 * - Query compatibility between types
 *
 * Usage:
 * @code
 * auto& registry = MetadataRegistry::getInstance();
 *
 * // Register conversion
 * registry.registerConversion(
 *     FrameMetadata::FACEDETECTS_INFO,
 *     FrameMetadata::OVERLAY_INFO_IMAGE,
 *     [](const frame_sp& face) -> frame_sp {
 *         // Convert face detections to overlay
 *         return convertedFrame;
 *     },
 *     1  // Cost
 * );
 *
 * // Check compatibility
 * if (registry.areCompatible(sourceType, targetType)) {
 *     frame_sp converted = registry.convertFrame(sourceFrame, targetType);
 * }
 * @endcode
 */
class MetadataRegistry {
public:
    /**
     * @brief Get singleton instance
     */
    static MetadataRegistry& getInstance();

    /**
     * @brief Register a conversion from source to target type
     *
     * @param source Source metadata type
     * @param target Target metadata type
     * @param converter Conversion function
     * @param cost Cost of this conversion (lower is better, typically 1)
     */
    void registerConversion(
        FrameMetadata::FrameType source,
        FrameMetadata::FrameType target,
        ConverterFunc converter,
        int cost = 1
    );

    /**
     * @brief Find conversion path from source to target type
     *
     * Uses Dijkstra's algorithm to find shortest path.
     *
     * @param source Source metadata type
     * @param target Target metadata type
     * @return Conversion path if found, empty optional otherwise
     */
    std::optional<ConversionPath> findConversionPath(
        FrameMetadata::FrameType source,
        FrameMetadata::FrameType target
    ) const;

    /**
     * @brief Convert frame to target metadata type
     *
     * Finds conversion path and applies all steps.
     *
     * @param sourceFrame Source frame
     * @param targetType Target metadata type
     * @return Converted frame, or nullptr if conversion not possible
     */
    frame_sp convertFrame(
        const frame_sp& sourceFrame,
        FrameMetadata::FrameType targetType
    ) const;

    /**
     * @brief Check if two types are compatible (convertible)
     *
     * @param source Source metadata type
     * @param target Target metadata type
     * @return true if conversion path exists, false otherwise
     */
    bool areCompatible(
        FrameMetadata::FrameType source,
        FrameMetadata::FrameType target
    ) const;

    /**
     * @brief Get all types that source can convert to
     *
     * @param source Source metadata type
     * @return Vector of compatible target types
     */
    std::vector<FrameMetadata::FrameType> getCompatibleOutputTypes(
        FrameMetadata::FrameType source
    ) const;

    /**
     * @brief Get all registered conversions for debugging
     *
     * @return Map of source types to their registered target conversions
     */
    std::map<FrameMetadata::FrameType, std::vector<FrameMetadata::FrameType>>
    getRegisteredConversions() const;

    /**
     * @brief Clear all registered conversions (mainly for testing)
     */
    void clear();

private:
    // Singleton pattern
    MetadataRegistry() = default;
    ~MetadataRegistry() = default;
    MetadataRegistry(const MetadataRegistry&) = delete;
    MetadataRegistry& operator=(const MetadataRegistry&) = delete;

    // Storage: map from source type to list of possible conversions
    std::map<FrameMetadata::FrameType, std::vector<ConversionStep>> mConversions;
};

/**
 * @brief Initialize all built-in metadata conversions
 *
 * Call this once at application startup to register standard conversions.
 * Typically called from PipeLine initialization or main().
 */
void registerBuiltinConversions();
