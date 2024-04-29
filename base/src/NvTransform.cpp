#include "NvTransform.h"
#include "AIPExceptions.h"
#include "DMAAllocator.h"
#include "DMAFDWrapper.h"
#include "Frame.h"
#include "FrameMetadata.h"
#include "Logger.h"
#include "Utils.h"
#include "nvbuf_utils.h"

#include "npp.h"

/**
 * @brief The Detail class for NvTransform Module, contains internal
 * implementation details.
 */
class NvTransform::Detail {
public:
  /**
   * @brief Constructor to initialize the Detail object with given properties.
   * @param _props The NvTransformProps object containing transformation
   * properties.
   */
  Detail(NvTransformProps &_props) : props(_props) {
    memset(&transParams, 0, sizeof(transParams));
    setSourceRect(_props, transParams);
    setDestinationRect(_props, transParams);
    setFilterType(_props, transParams);
    setRotateType(_props, transParams);
  }

  ~Detail() {}

  /**
   * @brief Computes the transformation and writes the result to the output
   * frame.
   * @param frame The input frame to be transformed.
   * @param outFD The file descriptor of the output frame.
   * @return True if computation is successful, false otherwise.
   */
  bool compute(frame_sp &frame, int outFD) {
    DMAFDWrapper *dmaFDWrapper = static_cast<DMAFDWrapper *>(frame->data());
    NvBufferTransform(dmaFDWrapper->getFd(), outFD, &transParams);
    return true;
  }

  /**
   * @brief Sets the transformation properties.
   * @param _props The new transformation properties.
   */
  void setProps(NvTransformProps &_props) {
    props = _props;
    updateTransformationParams(props);
  }

  /**
   * @brief Updates the transformation parameters based on new properties.
   * @param _props The new transformation properties.
   */
  void updateTransformationParams(NvTransformProps &_props) {
    memset(&transParams, 0, sizeof(transParams));
    setSourceRect(_props, transParams);
    setDestinationRect(_props, transParams);
    setFilterType(_props, transParams);
    setRotateType(_props, transParams);
  }

  /**
   * @brief Sets the source rectangle for transformation.
   * @param _props The transformation properties.
   * @param transParams The transformation parameters.
   */
  void setSourceRect(NvTransformProps &_props,
                     NvBufferTransformParams &transParams) {
    srcRect.top = _props.srcRect.top;
    srcRect.left = _props.srcRect.left;
    srcRect.width = _props.srcRect.width;
    srcRect.height = _props.srcRect.height;

    // Set source rectangle if width is non-zero
    if (srcRect.width != 0) {
      transParams.src_rect = srcRect;
      transParams.transform_flag |= NVBUFFER_TRANSFORM_CROP_SRC;
    }
  }

  /**
   * @brief Sets the destination rectangle for transformation.
   * @param _props The transformation properties.
   * @param transParams The transformation parameters.
   */
  void setDestinationRect(NvTransformProps &_props,
                          NvBufferTransformParams &transParams) {
    dstRect.top = _props.dstRect.top;
    dstRect.left = _props.dstRect.left;
    dstRect.width = _props.dstRect.width;
    dstRect.height = _props.dstRect.height;

    // Set destination rectangle if width is non-zero
    if (dstRect.width != 0) {
      transParams.dst_rect = dstRect;
      transParams.transform_flag |= NVBUFFER_TRANSFORM_CROP_DST;
    }
  }

  /**
   * @brief Sets the filter type for transformation.
   * @param _props The transformation properties.
   * @param transParams The transformation parameters.
   */
  void setFilterType(NvTransformProps &_props,
                     NvBufferTransformParams &transParams) {
    switch (_props.filterType) {
    case NvTransformProps::NvTransformFilter::NEAREST:
      transParams.transform_filter = NvBufferTransform_Filter_Nearest;
      break;
    case NvTransformProps::NvTransformFilter::BILINEAR:
      transParams.transform_filter = NvBufferTransform_Filter_Bilinear;
      break;
    case NvTransformProps::NvTransformFilter::TAP_5:
      transParams.transform_filter = NvBufferTransform_Filter_5_Tap;
      break;
    case NvTransformProps::NvTransformFilter::TAP_10:
      transParams.transform_filter = NvBufferTransform_Filter_10_Tap;
      break;
    case NvTransformProps::NvTransformFilter::SMART:
      transParams.transform_filter = NvBufferTransform_Filter_Smart;
      break;
    case NvTransformProps::NvTransformFilter::NICEST:
      transParams.transform_filter = NvBufferTransform_Filter_Nicest;
      break;
    default:
      throw AIPException(AIP_FATAL, "Filter Not Supported");
    }

    transParams.transform_flag |= NVBUFFER_TRANSFORM_FILTER;
  }

  /**
   * @brief Sets the rotation type for transformation.
   * @param _props The transformation properties.
   * @param transParams The transformation parameters.
   */
  void setRotateType(NvTransformProps &_props,
                     NvBufferTransformParams &transParams) {
    switch (_props.rotateType) {
    case NvTransformProps::NvTransformRotate::ROTATE_0:
      transParams.transform_flip = NvBufferTransform_None;
      break;
    case NvTransformProps::NvTransformRotate::ANTICLOCKWISE_ROTATE_90:
      transParams.transform_flip = NvBufferTransform_Rotate90;
      break;
    case NvTransformProps::NvTransformRotate::ANTICLOCKWISE_ROTATE_180:
      transParams.transform_flip = NvBufferTransform_Rotate180;
      break;
    case NvTransformProps::NvTransformRotate::ANTICLOCKWISE_ROTATE_270:
      transParams.transform_flip = NvBufferTransform_Rotate270;
      break;
    case NvTransformProps::NvTransformRotate::CLOCKWISE_ROTATE_90:
      transParams.transform_flip = NvBufferTransform_Rotate270;
      break;
    case NvTransformProps::NvTransformRotate::CLOCKWISE_ROTATE_180:
      transParams.transform_flip = NvBufferTransform_Rotate180;
      break;
    case NvTransformProps::NvTransformRotate::CLOCKWISE_ROTATE_270:
      transParams.transform_flip = NvBufferTransform_Rotate90;
      break;
    default:
      throw AIPException(
          AIP_FATAL,
          "Rotate type Not Supported, please select from 0, 90, 180 or 270");
    }

    transParams.transform_flag |= NVBUFFER_TRANSFORM_FLIP;
  }

public:
  NvBufferRect srcRect, dstRect;   /**< Source and destination rectangles */
  framemetadata_sp outputMetadata; /**< Output frame metadata */
  std::string outputPinId;         /**< Output pin ID */
  NvTransformProps props;          /**< Transformation properties */

private:
  NvBufferTransformParams
      transParams; /**< NvBuffer Transformation parameters */
};

/**
 * @brief Constructor to initialize the NvTransform module with given
 * properties.
 * @param props The NvTransformProps object containing transformation
 * properties.
 */
NvTransform::NvTransform(NvTransformProps props)
    : Module(TRANSFORM, "NvTransform", props) {
  mDetail.reset(new Detail(props));
}

NvTransform::~NvTransform() {}

bool NvTransform::validateInputPins() {
  if (getNumberOfInputPins() != 1) {
    LOG_ERROR << "<" << getId()
              << ">::validateInputPins size is expected to be 1. Actual<"
              << getNumberOfInputPins() << ">";
    return false;
  }

  framemetadata_sp metadata = getFirstInputMetadata();
  FrameMetadata::FrameType frameType = metadata->getFrameType();
  if (frameType != FrameMetadata::RAW_IMAGE &&
      frameType != FrameMetadata::RAW_IMAGE_PLANAR) {
    LOG_ERROR << "<" << getId()
              << ">::validateInputPins input frameType is expected to be "
                 "RAW_IMAGE or RAW_IMAGE_PLANAR. Actual<"
              << frameType << ">";
    return false;
  }

  FrameMetadata::MemType memType = metadata->getMemType();
  if (memType != FrameMetadata::MemType::DMABUF) {
    LOG_ERROR << "<" << getId()
              << ">::validateInputPins input memType is expected to be DMABUF. "
                 "Actual<"
              << memType << ">";
    return false;
  }

  return true;
}

bool NvTransform::validateOutputPins() {
  if (getNumberOfOutputPins() != 1) {
    LOG_ERROR << "<" << getId()
              << ">::validateOutputPins size is expected to be 1. Actual<"
              << getNumberOfOutputPins() << ">";
    return false;
  }

  framemetadata_sp metadata = getFirstOutputMetadata();
  FrameMetadata::FrameType frameType = metadata->getFrameType();
  if (frameType != FrameMetadata::RAW_IMAGE &&
      frameType != FrameMetadata::RAW_IMAGE_PLANAR) {
    LOG_ERROR << "<" << getId()
              << ">::validateOutputPins input frameType is expected to be "
                 "RAW_IMAGE or RAW_IMAGE_PLANAR. Actual<"
              << frameType << ">";
    return false;
  }

  FrameMetadata::MemType memType = metadata->getMemType();
  if (memType != FrameMetadata::MemType::DMABUF) {
    LOG_ERROR << "<" << getId()
              << ">::validateOutputPins input memType is expected to be "
                 "DMABUF. Actual<"
              << memType << ">";
    return false;
  }

  return true;
}

void NvTransform::addInputPin(framemetadata_sp &metadata, string &pinId) {
  Module::addInputPin(metadata, pinId);
  switch (mDetail->props.imageType) {
  case ImageMetadata::BGRA:
  case ImageMetadata::RGBA:
    mDetail->outputMetadata =
        framemetadata_sp(new RawImageMetadata(FrameMetadata::MemType::DMABUF));
    break;
  case ImageMetadata::NV12:
  case ImageMetadata::YUV420:
    mDetail->outputMetadata = framemetadata_sp(
        new RawImagePlanarMetadata(FrameMetadata::MemType::DMABUF));
    break;
  default:
    throw AIPException(AIP_FATAL, "Unsupported Image Type<" +
                                      std::to_string(mDetail->props.imageType) +
                                      ">");
  }

  mDetail->outputMetadata->copyHint(*metadata.get());
  mDetail->outputPinId = addOutputPin(mDetail->outputMetadata);
}

bool NvTransform::init() {
  if (!Module::init()) {
    return false;
  }

  return true;
}

bool NvTransform::term() { return Module::term(); }

bool NvTransform::process(frame_container &frames) {
  frame_sp frame = frames.cbegin()->second;
  frame_sp outFrame =
      makeFrame(mDetail->outputMetadata->getDataSize(), mDetail->outputPinId);
  if (!outFrame.get()) {
    LOG_ERROR << "FAILED TO GET BUFFER";
    return false;
  }

  DMAFDWrapper *dmaFdWrapper = static_cast<DMAFDWrapper *>(outFrame->data());
  dmaFdWrapper->tempFD = dmaFdWrapper->getFd();

  mDetail->compute(frame, dmaFdWrapper->tempFD);

  frames.insert(make_pair(mDetail->outputPinId, outFrame));
  send(frames);

  return true;
}

bool NvTransform::processSOS(frame_sp &frame) {
  framemetadata_sp metadata = frame->getMetadata();
  setMetadata(metadata);
  return true;
}

void NvTransform::setMetadata(framemetadata_sp &metadata) {
  FrameMetadata::FrameType frameType = metadata->getFrameType();
  int width = 0;
  int height = 0;
  int depth = CV_8U;
  ImageMetadata::ImageType inputImageType = ImageMetadata::ImageType::MONO;

  switch (frameType) {
  case FrameMetadata::FrameType::RAW_IMAGE: {
    auto rawMetadata =
        FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
    width = rawMetadata->getWidth();
    height = rawMetadata->getHeight();
    depth = rawMetadata->getDepth();
    inputImageType = rawMetadata->getImageType();
  } break;
  case FrameMetadata::FrameType::RAW_IMAGE_PLANAR: {
    auto rawMetadata =
        FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
    width = rawMetadata->getWidth(0);
    height = rawMetadata->getHeight(0);
    depth = rawMetadata->getDepth();
    inputImageType = rawMetadata->getImageType();
  } break;
  default:
    throw AIPException(AIP_NOTIMPLEMENTED, "Unsupported FrameType<" +
                                               std::to_string(frameType) + ">");
  }

  if (mDetail->props.outputWidth != 0) {
    width = mDetail->props.outputWidth;
    height = mDetail->props.outputHeight;
  }
  if (mDetail->props.scaleHeight != 0 && mDetail->props.scaleWidth != 0) {
    width = width * mDetail->props.scaleWidth;
    height = height * mDetail->props.scaleHeight;
  }

  DMAAllocator::setMetadata(mDetail->outputMetadata, width, height,
                            mDetail->props.imageType);
}

bool NvTransform::processEOS(string &pinId) {
  mDetail->outputMetadata.reset();
  return true;
}

void NvTransform::setProps(NvTransformProps &props) {
  Module::addPropsToQueue(props);
}

NvTransformProps NvTransform::getProps() {
  fillProps(mDetail->props);
  return mDetail->props;
}

bool NvTransform::handlePropsChange(frame_sp &frame) {
  NvTransformProps props(mDetail->props.imageType);
  bool ret = Module::handlePropsChange(frame, props);
  mDetail->setProps(props);
  return ret;
}