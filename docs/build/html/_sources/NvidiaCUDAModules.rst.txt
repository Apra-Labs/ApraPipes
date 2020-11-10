NVIDIA CUDA Modules
===================

Getting Started
^^^^^^^^^^^^^^^

Nvidia CUDA Modules
^^^^^^^^^^^^^^^^^^^

cudaStream_t
""""""""""""
Checkout `CUDA DOCUMENTATION <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html>`_
and `CUDA WEBINAR <https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf>`_

- | A sequence of operations that execute in issue-order on the GPU
- | Programming model used to effect concurrency
- | CUDA operations in different streams may run concurrently
- | CUDA operations from different streams may be interleaved

cudaStream_t has to be passed as props to all the CUDA modules.
cudaStreamCreate and cudaStreamDestroy is mandatory.

CudaMemCopyProps
""""""""""""""""
- | cudaMemcpyKind memcpyKind - cudaMemcpyDeviceToHost

   - | Input is CUDA_DEVICE memory
   - | Output is HOST memory   
   - | sync is set to true internally 
- | cudaMemcpyKind memcpyKind - cudaMemcpyHostToDevice

   - | Input is HOST memory
   - | Output is CUDA_DEVICE memory
- | sync - if sync is true, stream will be synchronized - don't set it to true until you know what you are doing- | 
- | cudaStream_t has to be passed

JPEGDecoderNVJPEG
"""""""""""""""""
- | Input is HOST memory
- | Output is CUDA_DEVICE memory
- | Output ImageMetadata::ImageType - MONO, YUV420, YUV444 is currently supported
- | cudaStream_t has to be passed

JPEGEncoderNVJPEG
"""""""""""""""""
- | Input is CUDA_DEVICE memory
- | Output is HOST memory
- | quality - default is 90
- | Input ImageMetadata::ImageType - MONO, YUV420, YUV444 is currently supported
- | cudaStream_t has to be passed

ResizeNPPI
""""""""""
- | Input is CUDA_DEVICE memory
- | Output is CUDA_DEVICE memory
- | output width and height has to be passed
- | Input ImageMetadata::ImageType - MONO, YUV444, YUV420, BGR, BGRA, RGB, RGBA
- | cudaStream_t has to be passed

CCNPPI
""""""
- | Input is CUDA_DEVICE memory
- | Output is CUDA_DEVICE memory
- | Supported Color Conversions

   - | MONO, YUV420 To BGRA
   - | YUV420 To BGRA
   - | YUV411_I To YUV444
- | cudaStream_t has to be passed

OverlayNPPI
"""""""""""
- | Input is CUDA_DEVICE memory
- | Output is CUDA_DEVICE memory
- | Input ImageMetadata::ImageType - BGRA/RGBA or YUV420
- | Accepts two images - Source and Overlay
- | OVERLAY_HINT has to be set to tell the module which of the two input images is the overlay image
- | Overlay image width and height is expected to be less than or equal to the Source image
- | offsetX and offsetY can be given
- | cudaStream_t has to be passed

EffectsNPPI
"""""""""""
- | Input is CUDA_DEVICE memory
- | Output is CUDA_DEVICE memory
- | Input ImageMetadata::ImageType - MONO/BGR/RGB/RGBA/BGRA/YUV420 
- | cudaStream_t has to be passed

.. code-block:: c++

    double hue;            // 0/255 no change [0 255] 
    double saturation;     // 1 no change
    double contrast;       // 1 no change 
    int brightness;     // 0 no change [-100 - 100]

H264EncoderNVCodec
""""""""""""""""""
- | Input is CUDA_DEVICE memory
- | Output is HOST memory
- | Input ImageMetadata::ImageType - BGRA/RGBA or YUV420
- | targetKbps can be passed
- | refer test - h264Encodernvcodec_tests
- | apracucontext_sp must be passed
- | Initialize CudaContext before constructing the pipeline.  `auto cuContext = apracucontext_sp(new ApraCUcontext());`
- | cudaStreamSynchronize has to be called before this module. Use one of the below methods. 

      - | sync = true for CudaMemCopyProps
      - | Add CudaStreamSynchronize Module

GaussianBlur
""""""""""""
- | Input is CUDA_DEVICE memory
- | Output is CUDA_DEVICE memory
- | Props

      - | cudastream_sp
      - | kernel size
      - | ROI

NonmaxSuppression
"""""""""""""""""
Sobel followed by nonmax suppression is applied.

- | kernel size of sobel/scharr operator is hardcoded to 3 inside the cuda kernel
- | ideally all the parameters of opencv should be supported https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga04723e007ed888ddf11d9ba04e2232de


- | Input is CUDA_DEVICE memory
- | Output is CUDA_DEVICE memory 
   - | Nonmax image
   - | Phase image
   - | Magnitude image can also be sent out with very minor changes
- | Props

      - | cudastream_sp
      - | ROI

HysteresisThreshold
"""""""""""""""""""
- | Input is CUDA_DEVICE memory
- | Output is CUDA_DEVICE memory
- | Props

      - | cudastream_sp
      - | ROI
      - | Low Threshold
      - | High Threshold
      - | Low Phase Threshold
      - | High Phase Threshold

HoughLinesCV
""""""""""""
- | Input is CUDA_DEVICE memory
- | Output is HOST memory of type APRA_LINES
- | cudaStreamSynchronize is automatically called in this module
- | Props

      - | cudastream_sp
      - | ROI
      - | Minimum Line Length
      - | Maximum Line Gap

Building Pipelines
^^^^^^^^^^^^^^^^^^

Example 1
"""""""""
.. image:: nvidiacudamodules_1.jpg

.. code-block:: c++

   auto width = 1920;
   auto height = 1080;

   auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/mono_1920x1080.raw")));
   auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::MONO, CV_8UC1, 0, CV_8U, FrameMetadata::HOST, true));
   fileReader->addOutputPin(metadata);

   cudaStream_t stream;
   cudaStreamCreate(&stream);
   auto copy = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
   fileReader->setNext(copy);  

   auto encoder = boost::shared_ptr<JPEGEncoderNVJPEG>(new JPEGEncoderNVJPEG(JPEGEncoderNVJPEGProps(stream)));
   copy->setNext(encoder);

   auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule("./data/testOutput/mono_1920x1080.jpg"));
   encoder->setNext(fileWriter);

   ...
   
   cudaStreamDestroy(stream);

Example 2
"""""""""
.. image:: nvidiacudamodules_2.jpg

.. code-block:: c++

   auto fileReader = boost::shared_ptr<Module>(new FileReaderModule(FileReaderModuleProps("./data/mono_1920x960.jpg")));
   auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
   fileReader->addOutputPin(metadata);

   cudaStream_t stream;
   cudaStreamCreate(&stream);

   auto decoder = boost::shared_ptr<JPEGDecoderNVJPEG>(new JPEGDecoderNVJPEG(JPEGDecoderNVJPEGProps(stream)));
   fileReader->setNext(decoder);

   auto resize = boost::shared_ptr<Module>(new ResizeNPPI(ResizeNPPIProps(960, 480, stream)));
   decoder->setNext(resize);

   auto copy = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyDeviceToHost, stream)));
   resize->setNext(copy);

   auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule("./data/testOutput/mono_1920x960_to_mono_960x480.raw"));
   encoder->setNext(fileWriter);
   
   ...
   
   cudaStreamDestroy(stream);

Example 3
"""""""""
.. image:: nvidiacudamodules_3.jpg

.. code-block:: c++

   cudaStream_t stream;
   cudaStreamCreate(&stream);

   // making source pipeline - JPEGDecoderNVJPEG -> ResizeNPPI -> CCNPPI 
   auto fileReader = boost::shared_ptr<Module>(new FileReaderModule(FileReaderModuleProps("./data/mono_1920x960.jpg")));
   auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
   fileReader->addOutputPin(metadata);

   auto decoder = boost::shared_ptr<JPEGDecoderNVJPEG>(new JPEGDecoderNVJPEG(JPEGDecoderNVJPEGProps(stream)));
   fileReader->setNext(decoder);

   auto resize = boost::shared_ptr<Module>(new ResizeNPPI(ResizeNPPIProps(960, 480, stream)));
   decoder->setNext(resize);

   auto source_cc = boost::shared_ptr<Module>(new CCNPPI(CCNPPIProps(ImageMetadata::BGRA, stream)));
   resize->setNext(source_cc);

   // making overlay pipeline - CudaMemCopy -> ResizeNPPI
   auto overlay_host = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/overlay_1920x960_BGRA.raw")));
   auto overlay_metadata = framemetadata_sp(new RawImageMetadata(1920, 960, ImageMetadata::ImageType::BGRA, CV_8UC4, 0, CV_8U, FrameMetadata::HOST, true));	
   overlay_metadata->setHint(OVERLAY_HINT);
   overlay_host->addOutputPin(overlay_metadata);

   auto overlay_copy = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
   overlay_host->setNext(overlay_copy);	

   auto overlay_resize = boost::shared_ptr<Module>(new ResizeNPPI(ResizeNPPIProps(960, 480, stream)));
   overlay_copy->setNext(overlay_resize);

   // Connecting source and overlay pipelines
   auto overlay = boost::shared_ptr<Module>(new OverlayNPPI(OverlayNPPIProps(stream)));
   overlay_resize->setNext(overlay);
   source_cc->setNext(overlay);

   auto encoder = boost::shared_ptr<Module>(new JPEGEncoderNVJPEG(JPEGEncoderNVJPEGProps(stream)));
   overlay->setNext(encoder);

   auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule("./data/testOutput/out.jpg"));
   encoder->setNext(fileWriter);
   
   ...
   
   cudaStreamDestroy(stream);