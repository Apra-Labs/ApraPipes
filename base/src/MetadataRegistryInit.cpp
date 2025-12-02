#include "MetadataRegistry.h"
#include "FaceDetectsInfo.h"
#include "Overlay.h"
#include "Logger.h"
#include "FrameMetadataFactory.h"

/**
 * @brief Register all built-in metadata conversions
 *
 * This function is called once at application startup to register
 * standard metadata type conversions.
 */
void registerBuiltinConversions() {
    auto& registry = MetadataRegistry::getInstance();

    LOG_INFO << "Registering built-in metadata conversions...";

    // ========================================================================
    // FACEDETECTS_INFO → OVERLAY_INFO_IMAGE
    // ========================================================================
    // Converts face detection results to overlay visualization with:
    // - Green rectangles around detected faces
    // - White text showing confidence score (percentage)
    registry.registerConversion(
        FrameMetadata::FACEDETECTS_INFO,
        FrameMetadata::OVERLAY_INFO_IMAGE,
        [](const frame_sp& faceFrame) -> frame_sp {
            if (!faceFrame) {
                LOG_ERROR << "FaceDetects→Overlay: Null frame";
                return nullptr;
            }

            // Deserialize face detection results
            FaceDetectsInfo faceInfo;
            Utils::deSerialize<FaceDetectsInfo>(faceInfo, faceFrame->data(), faceFrame->size());

            if (!faceInfo.facesFound || faceInfo.faces.empty()) {
                LOG_TRACE << "FaceDetects→Overlay: No faces detected";
                // Return empty overlay
                DrawingOverlay emptyOverlay;
                size_t overlaySize = emptyOverlay.mGetSerializeSize();

                // Allocate buffer for output frame
                void* buffer = operator new(overlaySize);

                // Create output frame with allocated buffer
                auto metadata = framemetadata_sp(new FrameMetadata(
                    FrameMetadata::OVERLAY_INFO_IMAGE));
                // Note: passing nullptr for factory means frame won't be pooled
                // This is acceptable for metadata conversions which are typically small
                auto outFrame = frame_sp(new Frame(buffer, overlaySize, nullptr));
                outFrame->setMetadata(metadata);

                emptyOverlay.serialize(outFrame);
                return outFrame;
            }

            // Create composite overlay with rectangles and text
            DrawingOverlay drawing;

            for (const auto& face : faceInfo.faces) {
                // Convert each face to overlay using ApraFaceInfo's conversion
                size_t convertedSize = 0;
                void* converted = face.convertTo(FrameMetadata::OVERLAY_INFO_IMAGE, convertedSize);

                if (!converted) {
                    LOG_ERROR << "FaceDetects→Overlay: Failed to convert face info";
                    continue;
                }

                // The converted result is a DrawingOverlay*
                DrawingOverlay* faceOverlay = static_cast<DrawingOverlay*>(converted);

                // Add all components from face overlay to main drawing
                for (auto* component : faceOverlay->getList()) {
                    drawing.add(component);
                }

                // Note: We don't delete faceOverlay here because drawing now owns the components
                // The components will be cleaned up when drawing is destroyed
            }

            // Calculate output size
            size_t overlaySize = drawing.mGetSerializeSize();

            // Allocate buffer for output frame
            void* buffer = operator new(overlaySize);

            // Create output frame with OVERLAY_INFO_IMAGE metadata
            auto metadata = framemetadata_sp(new FrameMetadata(
                FrameMetadata::OVERLAY_INFO_IMAGE));
            auto outFrame = frame_sp(new Frame(buffer, overlaySize, nullptr));
            outFrame->setMetadata(metadata);

            // Serialize overlay into frame
            drawing.serialize(outFrame);

            LOG_DEBUG << "FaceDetects→Overlay: Converted " << faceInfo.faces.size()
                      << " faces to overlay";

            return outFrame;
        },
        1  // Cost: direct conversion
    );

    // ========================================================================
    // Future conversions can be added here:
    // ========================================================================

    // Example: FACEDETECTS_INFO → ROI_METADATA
    // registry.registerConversion(
    //     FrameMetadata::FACEDETECTS_INFO,
    //     FrameMetadata::ROI_METADATA,
    //     [](const frame_sp& face) -> frame_sp { ... },
    //     1
    // );

    // Example: ROI_METADATA → OVERLAY_INFO_IMAGE
    // registry.registerConversion(
    //     FrameMetadata::ROI_METADATA,
    //     FrameMetadata::OVERLAY_INFO_IMAGE,
    //     [](const frame_sp& roi) -> frame_sp { ... },
    //     1
    // );

    // With both conversions above registered, the system can automatically
    // convert FACEDETECTS_INFO → ROI_METADATA → OVERLAY_INFO_IMAGE
    // choosing the direct path (cost 1) over the two-hop path (cost 2)

    LOG_INFO << "Built-in metadata conversions registered successfully";
}
