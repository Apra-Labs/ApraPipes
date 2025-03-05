# **Color Conversion Module \- Knowledge Document**

## **Overview**

The `ColorConversion` module is responsible for converting images between different color formats using OpenCV. It supports various conversions such as RGB to BGR, RGB to MONO, RGB to YUV420, and Bayer to RGB.

## **Class Breakdown**

### **1\. `ColorConversion`**

This is the main class responsible for handling the color conversion process.

#### **Constructor and Destructor**

* **`ColorConversion(ColorConversionProps _props)`**: Initializes the module with transformation properties.  
* **`~ColorConversion()`**: Default destructor.

#### **Validation Methods**

* **`bool validateInputPins()`**: Ensures that the module receives exactly one input pin and the input is of type `RAW_IMAGE` or `RAW_IMAGE_PLANAR`.  
* **`bool validateOutputPins()`**: Ensures that the module produces exactly one output pin and the output is of type `RAW_IMAGE` or `RAW_IMAGE_PLANAR`.

#### **Initialization & Processing**

* **`bool init()`**: Initializes the module.  
* **`bool term()`**: Terminates the module.  
* **`bool process(frame_container& frames)`**: Processes frames by applying color conversion.  
* **`bool processSOS(frame_sp& frame)`**: Handles Start of Stream (SOS) processing.  
* **`bool shouldTriggerSOS()`**: Determines if SOS should be triggered.

#### **Metadata & Strategy**

* **`bool setMetadata(framemetadata_sp& metadata)`**: Configures metadata for input and output frames based on conversion type.  
* **`void setConversionStrategy(framemetadata_sp inputMetadata, framemetadata_sp outputMetadata)`**: Chooses the appropriate conversion strategy based on metadata.

---

### **2\. `DetailAbstract`**

Abstract class for color conversion strategies using OpenCV’s `cv::Mat`.

* **Attributes**:  
  * `cv::Mat inpImg`: Input image matrix.  
  * `cv::Mat outImg`: Output image matrix.  
* **Methods**:  
  * `virtual void convert(frame_container& inputFrame, frame_sp& outFrame, framemetadata_sp outputMetadata)`: Pure virtual method for conversion.

---

### **3\. Derived Conversion Classes**

These classes implement specific color conversion techniques.

#### **Interleaved to Planar Conversions**

* **`CpuInterleaved2Planar`**: Base class for interleaved to planar conversions.  
* **`CpuRGB2YUV420Planar`**: Converts RGB to YUV420 planar using OpenCV’s `cv::cvtColor(inpImg, outImg, cv::COLOR_RGB2YUV_I420)`.

#### **Interleaved to Interleaved Conversions**

* **`CpuInterleaved2Interleaved`**: Base class for interleaved to interleaved conversions.  
* **`CpuRGB2BGR`**: Converts RGB to BGR.  
* **`CpuBGR2RGB`**: Converts BGR to RGB.  
* **`CpuRGB2MONO`**: Converts RGB to grayscale (MONO).  
* **`CpuBGR2MONO`**: Converts BGR to grayscale (MONO).

#### **Bayer to RGB/MONO Conversions**

* **`CpuBayerBG82RGB`**: Converts Bayer BG8 to RGB.  
* **`CpuBayerGB82RGB`**: Converts Bayer GB8 to RGB.  
* **`CpuBayerGR82RGB`**: Converts Bayer GR8 to RGB.  
* **`CpuBayerRG82RGB`**: Converts Bayer RG8 to RGB.  
* **`CpuBayerBG82Mono`**: Converts Bayer BG8 to grayscale.

#### **Planar to Interleaved Conversions**

* **`CpuPlanar2Interleaved`**: Base class for planar to interleaved conversions.  
* **`CpuYUV420Planar2RGB`**: Converts YUV420 planar to RGB using OpenCV’s `cv::cvtColor(inpImg, outImg, cv::COLOR_YUV420p2RGB)`.

---

### **4\. `AbsColorConversionFactory`**

A factory class responsible for selecting the appropriate conversion strategy based on input and output metadata.

#### **Key Method**

* **`boost::shared_ptr<DetailAbstract> create(framemetadata_sp input, framemetadata_sp output, cv::Mat& inpImg, cv::Mat& outImg)`**  
  * Determines the type of color conversion required.  
  * Returns the appropriate conversion strategy class instance.  
  * Throws an exception if the conversion is not supported.

## **Summary of Supported Conversions**

| Input Format | Output Format | Conversion Class |
| :---- | :---- | :---- |
| RGB | BGR | CpuRGB2BGR |
| BGR | RGB | CpuBGR2RGB |
| RGB | MONO | CpuRGB2MONO |
| BGR | MONO | CpuBGR2MONO |
| RGB | YUV 420 Planar | CpuRGB2YUV420Planar |
| YUV420 Planar | RGB | CpuYUV420Planar2RGB |
| Bayer BG8 | RGB | CpuBayerBG82RGB |
| Bayer BG8 | MONO | CpuBayerBG82Mono |
| Bayer GB8 | RGB | CpuBayerGB82RGB |
| Bayer GR8 | RGB | CpuBayerGR82RGB |
| Bayer RG8 | RGB | CpuBayerRG82RGB |

## **Exception Handling**

The module uses `AIPException(AIP_FATAL, "conversion not supported")` to handle unsupported conversions, ensuring robustness.

## **Conclusion**

This module provides a flexible and efficient framework for handling various color space conversions using OpenCV, making it suitable for applications in image processing and computer vision.

