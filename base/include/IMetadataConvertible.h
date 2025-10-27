#pragma once

#include "FrameMetadata.h"
#include <vector>

/**
 * @brief Interface for metadata structures that can convert to other metadata types
 *
 * This interface enables automatic metadata type conversion in the pipeline framework.
 * Implementing classes declare which metadata types they can convert to and provide
 * conversion logic.
 *
 * Example usage:
 * @code
 * class ApraFaceInfo : public IMetadataConvertible {
 *     std::vector<FrameMetadata::FrameType> getConvertibleTypes() const override {
 *         return {FrameMetadata::OVERLAY_INFO_IMAGE};
 *     }
 *
 *     void* convertTo(FrameMetadata::FrameType targetType, size_t& outSize) const override {
 *         if (targetType == FrameMetadata::OVERLAY_INFO_IMAGE) {
 *             return createOverlayRepresentation(outSize);
 *         }
 *         return nullptr;
 *     }
 * };
 * @endcode
 *
 * This is Phase 1 of the Intelligent Pipeline Framework (see intelligent_pipeline_design.md)
 */
class IMetadataConvertible {
public:
    virtual ~IMetadataConvertible() {}

    /**
     * @brief Get list of metadata types this can convert to
     * @return Vector of FrameMetadata::FrameType values
     */
    virtual std::vector<FrameMetadata::FrameType> getConvertibleTypes() const = 0;

    /**
     * @brief Convert to specified metadata type
     * @param targetType The desired metadata type
     * @param outSize Output parameter for size of returned data
     * @return Pointer to converted data (caller owns memory), or nullptr if conversion not supported
     */
    virtual void* convertTo(FrameMetadata::FrameType targetType, size_t& outSize) const = 0;

    /**
     * @brief Get the native metadata type of this data structure
     * @return The FrameMetadata::FrameType representing this data's native type
     */
    virtual FrameMetadata::FrameType getNativeType() const = 0;

    /**
     * @brief Check if conversion to a specific type is supported
     * @param targetType The type to check
     * @return true if conversion is supported, false otherwise
     */
    virtual bool canConvertTo(FrameMetadata::FrameType targetType) const {
        auto types = getConvertibleTypes();
        return std::find(types.begin(), types.end(), targetType) != types.end();
    }
};
