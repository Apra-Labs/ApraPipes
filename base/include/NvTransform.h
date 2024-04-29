#pragma once

#include "Module.h"

/**
 * @brief A class representing a rectangular crop region.
 *
 * This class encapsulates the parameters for defining a rectangular crop region
 * with top-left coordinates, width, and height.
 */
class CropRect : public ModuleProps {
public:
  /**
   * @brief Constructor for CropRect.
   *
   * Initializes a CropRect object with the specified parameters.
   *
   * @param _top The top coordinate of the crop region.
   * @param _left The left coordinate of the crop region.
   * @param _width The width of the crop region.
   * @param _height The height of the crop region.
   */
  CropRect(int _top, int _left, int _width, int _height)
      : top(_top), left(_left), width(_width), height(_height) {}

  int top;    /**< The top coordinate of the crop region. */
  int left;   /**< The left coordinate of the crop region. */
  int width;  /**< The width of the crop region. */
  int height; /**< The height of the crop region. */

  size_t getSerializeSize() {
    return ModuleProps::getSerializeSize() + sizeof(int) * 4;
  }

private:
  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive &ar, const unsigned int version) {
    ar &boost::serialization::base_object<ModuleProps>(*this);
    ar & top & left & width & height;
  }
};

/**
 * @class NvTransformProps
 * @brief Wrapper class for configuring Nvidia NvBuffer Transform Parameters
 */
class NvTransformProps : public ModuleProps {
public:
  /**
   * @enum NvTransformFilter
   * @brief Enumerates different types of transformation filters.
   */
  enum NvTransformFilter {
    NEAREST = 0, /**< Nearest neighbor interpolation. */
    BILINEAR,    /**< Bilinear interpolation. */
    TAP_5,       /**< 5-tap filter. */
    TAP_10,      /**< 10-tap filter. */
    SMART,       /**< Smart filter. */
    NICEST       /**< Nicest filter. */
  };

  /**
   * @enum NvTransformRotate
   * @brief Enumerates different types of rotation.
   */
  enum NvTransformRotate {
    ROTATE_0 = 0,             /**< No rotation. */
    ANTICLOCKWISE_ROTATE_90,  /**< Rotate 90 degrees anticlockwise. */
    ANTICLOCKWISE_ROTATE_180, /**< Rotate 180 degrees anticlockwise. */
    ANTICLOCKWISE_ROTATE_270, /**< Rotate 270 degrees anticlockwise. */
    CLOCKWISE_ROTATE_90,      /**< Rotate 90 degrees clockwise. */
    CLOCKWISE_ROTATE_180,     /**< Rotate 180 degrees clockwise. */
    CLOCKWISE_ROTATE_270      /**< Rotate 270 degrees clockwise. */
  };

  /**
   * @brief Constructs an NvTransformProps object.
   * @param _imageType The type of the input image.
   */
  NvTransformProps(ImageMetadata::ImageType _imageType)
      : outputWidth(0), outputHeight(0), srcRect({0, 0, 0, 0}),
        dstRect({0, 0, 0, 0}) {
    imageType = _imageType;
    scaleHeight = 1;
    scaleWidth = 1;
    filterType = NvTransformFilter::SMART;
    rotateType = NvTransformRotate::ROTATE_0;
  }

  /**
   * @brief Constructs an NvTransformProps object with a specified rotation
   * type.
   * @param _imageType The type of the input image.
   * @param _rotateType The rotation type.
   */
  NvTransformProps(ImageMetadata::ImageType _imageType,
                   NvTransformRotate _rotateType)
      : outputWidth(0), outputHeight(0), srcRect({0, 0, 0, 0}),
        dstRect({0, 0, 0, 0}) {
    imageType = _imageType;
    scaleHeight = 1;
    scaleWidth = 1;
    filterType = NvTransformFilter::SMART;
    rotateType = _rotateType;
  }

  /**
   * @brief Constructs an NvTransformProps object with a specified rotation
   * type.
   * @param _imageType The type of the input image.
   * @param _filterType The transformation filter type.
   * @param _rotateType The rotation type.
   */
  NvTransformProps(ImageMetadata::ImageType _imageType, NvTransformFilter _filterType, NvTransformRotate _rotateType)
      : outputWidth(0), outputHeight(0), srcRect({0, 0, 0, 0}),
        dstRect({0, 0, 0, 0}) {
    imageType = _imageType;
    scaleHeight = 1;
    scaleWidth = 1;
    filterType = _filterType;
    rotateType = _rotateType;
  }

  /**
   * @brief Constructs an NvTransformProps object with a specified width,
   * height, and rotation type.
   * @param _imageType The type of the input image.
   * @param _outputWidth The output width.
   * @param _outputHeight The output height.
   */
  NvTransformProps(ImageMetadata::ImageType _imageType, int _outputWidth,
                   int _outputHeight)
      : outputWidth(_outputWidth), outputHeight(_outputHeight),
        srcRect({0, 0, 0, 0}), dstRect({0, 0, 0, 0}) {
    imageType = _imageType;
    scaleHeight = 1;
    scaleWidth = 1;
    filterType = NvTransformFilter::SMART;
    rotateType = NvTransformRotate::ROTATE_0;
  }

  /**
   * @brief Constructs an NvTransformProps object with a specified width,
   * height, and rotation type.
   * @param _imageType The type of the input image.
   * @param _outputWidth The output width.
   * @param _outputHeight The output height.
   * @param _rotateType The rotation type.
   */
  NvTransformProps(ImageMetadata::ImageType _imageType, int _outputWidth,
                   int _outputHeight, NvTransformRotate _rotateType)
      : outputWidth(_outputWidth), outputHeight(_outputHeight),
        srcRect({0, 0, 0, 0}), dstRect({0, 0, 0, 0}) {
    imageType = _imageType;
    scaleHeight = 1;
    scaleWidth = 1;
    filterType = NvTransformFilter::SMART;
    rotateType = _rotateType;
  }

  /**
   * @brief Constructs an NvTransformProps object with specified parameters.
   * @param _imageType The type of the input image.
   * @param _outputWidth The output width.
   * @param _outputHeight The output height.
   * @param _filterType The transformation filter type.
   * @param _rotateType The rotation type.
   */
  NvTransformProps(ImageMetadata::ImageType _imageType, int _outputWidth,
                   int _outputHeight, NvTransformFilter _filterType,
                   NvTransformRotate _rotateType)
      : outputWidth(_outputWidth), outputHeight(_outputHeight),
        srcRect({0, 0, 0, 0}), dstRect({0, 0, 0, 0}) {
    imageType = _imageType;
    scaleHeight = 1;
    scaleWidth = 1;
    filterType = _filterType;
    rotateType = _rotateType;
  }

  /**
   * @brief Constructs an NvTransformProps object with specified parameters.
   * @param _imageType The type of the input image.
   * @param _outputWidth The output width.
   * @param _outputHeight The output height.
   * @param _scaleWidth The scaling factor for width.
   * @param _scaleHeight The scaling factor for height.
   * @param _filterType The transformation filter type.
   */
  NvTransformProps(ImageMetadata::ImageType _imageType, int _outputWidth,
                   int _outputHeight, float _scaleWidth, float _scaleHeight,
                   NvTransformFilter _filterType)
      : outputWidth(_outputWidth), outputHeight(_outputHeight),
        srcRect({0, 0, 0, 0}), dstRect({0, 0, 0, 0}) {
    imageType = _imageType;
    filterType = _filterType;
    scaleHeight = _scaleHeight;
    scaleWidth = _scaleWidth;
    rotateType = NvTransformRotate::ROTATE_0;
  }

  /**
   * @brief Constructs an NvTransformProps object with specified parameters.
   * @param _imageType The type of the input image.
   * @param _outputWidth The output width.
   * @param _outputHeight The output height.
   * @param _scaleWidth The scaling factor for width.
   * @param _scaleHeight The scaling factor for height.
   * @param _filterType The transformation filter type.
   * @param _rotateType The rotation type.
   */
  NvTransformProps(ImageMetadata::ImageType _imageType, int _outputWidth,
                   int _outputHeight, float _scaleWidth, float _scaleHeight,
                   NvTransformFilter _filterType, NvTransformRotate _rotateType)
      : outputWidth(_outputWidth), outputHeight(_outputHeight),
        srcRect({0, 0, 0, 0}), dstRect({0, 0, 0, 0}) {
    imageType = _imageType;
    filterType = _filterType;
    scaleHeight = _scaleHeight;
    scaleWidth = _scaleWidth;
    rotateType = _rotateType;
  }

  /**
   * @brief Constructs an NvTransformProps object with specified parameters.
   * @param _imageType The type of the input image.
   * @param _outputWidth The output width.
   * @param _outputHeight The output height.
   * @param _srcRect The source rectangle.
   */
  NvTransformProps(ImageMetadata::ImageType _imageType, int _outputWidth,
                   int _outputHeight, CropRect _srcRect)
      : outputWidth(_outputWidth), outputHeight(_outputHeight),
        srcRect(_srcRect), dstRect({0, 0, 0, 0}) {
    imageType = _imageType;
    scaleHeight = 1;
    scaleWidth = 1;
    filterType = NvTransformFilter::SMART;
    rotateType = NvTransformRotate::ROTATE_0;
  }

  /**
   * @brief Constructs an NvTransformProps object with specified parameters.
   * @param _imageType The type of the input image.
   * @param _outputWidth The output width.
   * @param _outputHeight The output height.
   * @param _srcRect The source rectangle.
   * @param _rotateType The rotation type.
   */
  NvTransformProps(ImageMetadata::ImageType _imageType, int _outputWidth,
                   int _outputHeight, CropRect _srcRect,
                   NvTransformRotate _rotateType)
      : outputWidth(_outputWidth), outputHeight(_outputHeight),
        srcRect(_srcRect), dstRect({0, 0, 0, 0}) {
    imageType = _imageType;
    scaleHeight = 1;
    scaleWidth = 1;
    filterType = NvTransformFilter::SMART;
    rotateType = _rotateType;
  }

  /**
   * @brief Constructs an NvTransformProps object with specified parameters.
   * @param _imageType The type of the input image.
   * @param _outputWidth The output width.
   * @param _outputHeight The output height.
   * @param _srcRect The source rectangle.
   * @param _scaleWidth The scaling factor for width.
   * @param _scaleHeight The scaling factor for height.
   * @param _filterType The transformation filter type.
   */
  NvTransformProps(ImageMetadata::ImageType _imageType, int _outputWidth,
                   int _outputHeight, CropRect _srcRect, float _scaleWidth,
                   float _scaleHeight, NvTransformFilter _filterType)
      : outputWidth(_outputWidth), outputHeight(_outputHeight),
        srcRect(_srcRect), dstRect({0, 0, 0, 0}) {
    imageType = _imageType;
    filterType = _filterType;
    scaleHeight = _scaleHeight;
    scaleWidth = _scaleWidth;
    rotateType = NvTransformRotate::ROTATE_0;
  }

  /**
   * @brief Constructs an NvTransformProps object with specified parameters.
   * @param _imageType The type of the input image.
   * @param _srcRect The source rectangle.
   * @param _dstRect The destination rectangle
   * @param _scaleWidth The scaling factor for width.
   * @param _scaleHeight The scaling factor for height.
   * @param _filterType The transformation filter type.
   * @param _rotateType The rotation type.
   */
  NvTransformProps(ImageMetadata::ImageType _imageType, CropRect _srcRect,
                   float _scaleWidth, float _scaleHeight,
                   NvTransformFilter _filterType, NvTransformRotate _rotateType)
      : outputWidth(0), outputHeight(0), srcRect(_srcRect),
        dstRect({0, 0, 0, 0}) {
    imageType = _imageType;
    filterType = _filterType;
    scaleHeight = _scaleHeight;
    scaleWidth = _scaleWidth;
    rotateType = _rotateType;
  }

  /**
   * @brief Constructs an NvTransformProps object with specified parameters.
   * @param _imageType The type of the input image.
   * @param _srcRect The source rectangle.
   * @param _filterType The transformation filter type.
   * @param _rotateType The rotation type.
   */
  NvTransformProps(ImageMetadata::ImageType _imageType, CropRect _srcRect,
                   NvTransformFilter _filterType, NvTransformRotate _rotateType)
      : outputWidth(0), outputHeight(0), srcRect(_srcRect),
        dstRect({0, 0, 0, 0}) {
    imageType = _imageType;
    filterType = _filterType;
    scaleHeight = 1;
    scaleWidth = 1;
    rotateType = _rotateType;
  }

  /**
   * @brief Constructs an NvTransformProps object with specified parameters.
   * @param _imageType The type of the input image.
   * @param _outputWidth The output width.
   * @param _outputHeight The output height.
   * @param _srcRect The source rectangle.
   * @param _filterType The transformation filter type.
   * @param _rotateType The rotation type.
   */
  NvTransformProps(ImageMetadata::ImageType _imageType, int _outputWidth,
                   int _outputHeight, CropRect _srcRect,
                   NvTransformFilter _filterType, NvTransformRotate _rotateType)
      : outputWidth(_outputWidth), outputHeight(_outputHeight),
        srcRect(_srcRect), dstRect({0, 0, 0, 0}) {
    imageType = _imageType;
    filterType = _filterType;
    scaleHeight = 1;
    scaleWidth = 1;
    rotateType = _rotateType;
  }

  /**
   * @brief Constructs an NvTransformProps object with specified parameters.
   * @param _imageType The type of the input image.
   * @param _outputWidth The output width.
   * @param _outputHeight The output height.
   * @param _srcRect The source rectangle.
   * @param _scaleWidth The scaling factor for width.
   * @param _scaleHeight The scaling factor for height.
   * @param _filterType The transformation filter type.
   * @param _rotateType The rotation type.
   */
  NvTransformProps(ImageMetadata::ImageType _imageType, int _outputWidth,
                   int _outputHeight, CropRect _srcRect, float _scaleWidth,
                   float _scaleHeight, NvTransformFilter _filterType,
                   NvTransformRotate _rotateType)
      : outputWidth(_outputWidth), outputHeight(_outputHeight),
        srcRect(_srcRect), dstRect({0, 0, 0, 0}) {
    imageType = _imageType;
    filterType = _filterType;
    scaleHeight = _scaleHeight;
    scaleWidth = _scaleWidth;
    rotateType = _rotateType;
  }

  /**
   * @brief Constructs an NvTransformProps object with specified parameters.
   * @param _imageType The type of the input image.
   * @param _outputWidth The output width.
   * @param _outputHeight The output height.
   * @param _srcRect The source rectangle.
   * @param _dstRect The destination rectangle
   * @param _scaleWidth The scaling factor for width.
   * @param _scaleHeight The scaling factor for height.
   * @param _filterType The transformation filter type.
   * @param _rotateType The rotation type.
   */
  NvTransformProps(ImageMetadata::ImageType _imageType, int _outputWidth,
                   int _outputHeight, CropRect _srcRect, CropRect _dstRect,
                   float _scaleWidth, float _scaleHeight,
                   NvTransformFilter _filterType, NvTransformRotate _rotateType)
      : outputWidth(_outputWidth), outputHeight(_outputHeight),
        srcRect(_srcRect), dstRect(_dstRect) {
    imageType = _imageType;
    filterType = _filterType;
    scaleHeight = _scaleHeight;
    scaleWidth = _scaleWidth;
    rotateType = _rotateType;
  }

  ImageMetadata::ImageType imageType; /**< The type of the input image. */
  int outputWidth, outputHeight;      /**< The output width and height. */
  NvTransformFilter filterType;       /**< The type of transformation filter. */
  float scaleWidth,
      scaleHeight; /**< The scaling factors for outputWidth and outputHeight,
                      otherwise the scale is done on the width and height read
                      from the metadata*/
  NvTransformRotate rotateType; /**< The rotation type. */
  CropRect srcRect, dstRect;    /**< Source and destination rectangles. */

  size_t getSerializeSize() {
    return ModuleProps::getSerializeSize() + sizeof(int) * 2 +
           sizeof(float) * 2 + sizeof(srcRect) * 2 + sizeof(imageType) +
           sizeof(filterType) + sizeof(rotateType);
  }

private:
  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive &ar, const unsigned int version) {
    ar &boost::serialization::base_object<ModuleProps>(*this);
    ar & outputWidth & outputHeight;
    ar & scaleWidth & scaleHeight;
    ar & srcRect & dstRect;
    ar & imageType;
    ar & filterType;
    ar & rotateType;
  }
};

/**
 * @class NvTransform
 * @brief Module class for performing transformations using NvBuffer Transform.
 */
class NvTransform : public Module {

public:
  NvTransform(NvTransformProps _props);
  virtual ~NvTransform();
  bool init();
  bool term();
  void setProps(NvTransformProps &props);
  NvTransformProps getProps();

protected:
  bool process(frame_container &frames);
  bool processSOS(frame_sp &frame);
  bool validateInputPins();
  bool validateOutputPins();
  void addInputPin(framemetadata_sp &metadata,
                   string &pinId); // throws exception if validation fails
  bool processEOS(string &pinId);
  bool handlePropsChange(frame_sp &frame);

private:
  void setMetadata(framemetadata_sp &metadata);
  class Detail;
  boost::shared_ptr<Detail> mDetail;
};