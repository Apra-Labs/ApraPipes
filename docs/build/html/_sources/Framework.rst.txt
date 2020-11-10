Framework
=========

ModuleProps
^^^^^^^^^^^
Base Properties class for all modules

.. code-block:: c++
   
	int fps; // can be updated during runtime with setProps
	size_t qlen; // run time changing doesn't effect this
	bool logHealth; // can be updated during runtime with setProps
	int logHealthFrequency; // 1000 by default - logs the health stats frequency

	// used for VimbaSource where we want to create the max frames and keep recycling it
	// for the VimbaDrive we announce frames after init - 100/200 
	// see VimbaSource.cpp on how it is used
	size_t maxConcurrentFrames; 

	// 0/1 - skipN == 0 -  don't skip any - process all
	// 1/1 - skipN == skipD - skip all - don't process any
	// 1/2 skips every alternate frame
	// 1/3 skips 1 out of every 3 frames
	// 2/3 skips 2 out of every 3 frames
	// 5/6 skips 5 out of every 6 frames
	// skipD >= skipN
	int skipN = 0; 
	int skipD = 1; 

QuePushStrategyType
^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

   enum QuePushStrategyType {		
	BLOCKING,
	NON_BLOCKING_ANY, // try push independently to child modules
	NON_BLOCKING_ALL_OR_NONE // try push only if all the child modules have the que free or don't push to any of the modules
   };
 
Module
^^^^^^
Base class for all the modules	

.. code-block:: c++
   
   void addOutputPin(framemetadata_sp& metadata, string& pinId);
   bool setNext(boost::shared_ptr<Module> next);   	

Pipeline
^^^^^^^^

.. code-block:: c++

   PipeLine(string name)

   /*
      call appendModule on only the topmost module - source (camera/file)
      call appendModule only after connecting all the modules - 
      if any modules are connected (module1->setNext(module2)) after appendModule then these modules will not be added to the pipeline 
      best practice is to first make the connections - all modules - then call appendModule
   */
   bool appendModule(boost::shared_ptr<Module> pModule);
   bool init();
   void run_all_threaded();
   
   void stop();   
   void term();
   void wait_for_all();


ExternalSourceModule
^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

   std::pair<bool, uint64_t> produceExternalFrame(ApraData* data)

- ``locked`` is atomically incremented and decremented
- std::pair<bool, uint64_t> is returned
- if data->fIndex == 0 fIndex is incremented by the framework and returns the value
- if data->fIndex != 0 fIndex is same as data->fIndex
- if ``false`` is returned, then the data is not accepted.Framework Que is full and there is no bandwidth to process more data
- it is upto the caller application to act on the response of ``produceExternalFrame``

.. code-block:: c++

   bool stop() 
   when ExternalSourceModule is not attached to the pipeline, manually call stop to trigger a tear down of the downstream modules


FileReaderModuleProps
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

   FileReaderModuleProps(const std::string& strFullFileNameWithPattern, int startIndex = 0, int maxIndex = -1, size_t maxFileSize =10000);

- loops till the ``maxIndex`` is reached
  ``-1`` loop the entire pattern/directory
- starts the loop from ``startIndex``

FileReaderModule
^^^^^^^^^^^^^^^^

.. code-block:: c++

   FileReaderModule(FileReaderModuleProps props);


JPEGDecoderIMProps
^^^^^^^^^^^^^^^^^^

.. code-block:: c++

   JPEGDecoderIMProps(bool sw);

- ``true`` if deocde is on software
- ``false`` if decode is on hardware

JPEGDecoderIM
^^^^^^^^^^^^^
.. code-block:: c++

   JPEGDecoderIM(JPEGDecoderIMProps props);

- Currently only supports single channel output

CalcHistogramCVProps
^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

   CalcHistogramCVProps(int bins);
   // All the properties can be updated during run time using setProps
   int bins;
   vector<int> roi;
   string maskImgPath;	
   
- ``bins`` number of histogram bins
- | ``maskImgPath`` absolute path of mask image path
  | Expected to be Same resolution as the dataset
  | if both roi and mask_img_path is specified roi is used and mask_img_path ignored
- | ``roi`` topleft_x topleft_y width height
  | if both roi and mask_img_path is specified roi is used and mask_img_path ignored

CalcHistogramCV
^^^^^^^^^^^^^^^

.. code-block:: c++
   
   CalcHistogramCV(CalcHistogramCVProps props);	
   CalcHistogramCVProps getProps();
   void setProps(CalcHistogramCVProps props);

   // Example on how to change Properties dynamically   
   auto histogram = new CalcHistogramCV(CalcHistogramCVProps(32));
   auto newHistProps = histogram->getProps();
   newHistProps.bins = newHistProps.bins = 16;
   histogram->setProps(newHistProps);


ChangeDetectionProps
^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

   ChangeDetectionProps(int refWindoweLength, int refDelayLength, int insWindowLength, double threshold, int compareMethod);
   // All the properties can be updated during run time using setProps

- | ``refWindowLength`` Number of frames to average
  | Assume ``fps = 30`` and you want to average ``2`` seconds then specify ``refWindowLength`` as ``60``
- | ``refDelayLength`` Number of frames to delay from the current frame 
  | default ``-1`` - Static reference
  | if moving average is required speicfy the delay 
  | Assume ``fps = 30`` and you want to delay 2 minutes then specify ``refDelayLength`` as ``3600``
- | ``insWindowLength`` Number of frames to average
  | Assume ``fps = 30`` and you want to average 0.5 seconds then specify ``insWindowLength`` as ``15``
- | ``compareMethod`` https://docs.opencv.org/4.1.1/d8/dc8/tutorial_histogram_comparison.html
  | HISTCMP_CHISQR (1) works for all the datasets provided

ChangeDetection
^^^^^^^^^^^^^^^
Sends ChangeDetectionResult to the next module

.. code-block:: c++
   
   ChangeDetection(ChangeDetectionProps props);
   ChangeDetectionProps getProps();
   void setProps(ChangeDetectionProps props);

   // Example on how to change Properties dynamically   
   auto module = new ChangeDetection(ChangeDetectionProps(60, -1, 30, 1, 1));
   auto newProps = module->getProps();
   newProps.threshold = 2;
   module->setProps(newProps);

JPEGEncoderIMProps
^^^^^^^^^^^^^^^^^^

.. code-block:: c++
   
   bool sw; // software or hardware mode
   unsigned short quality; // range 1-100 JPEG Compression quality - 100 means best quality but more file size

JPEGEncoderIM
^^^^^^^^^^^^^
Intel Media SDK based encoder

.. code-block:: c++
   
   JPEGEncoderIM(JPEGEncoderProps props);

- Currently only supports single channel as input

FileWriterModuleProps
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

   FileWriterModuleProps(const std::string& strFullFileNameWithPattern);

   // Example
   auto props = new FileWriterModuleProps("C:/Users/developer/Downloads/Sai/temp/enc/frame_????.jpg");


FileWriterModule
^^^^^^^^^^^^^^^^
Basic FileWriter

.. code-block:: c++

   FileWriterModule(FileWriterModuleProps props);

JPEGDecoderL4TM
^^^^^^^^^^^^^^^

.. code-block:: c++

   JPEGDecoderL4TMProps();
   JPEGDecoderL4TM(JPEGDecoderL4TMProps props);

- Currently only supports single channel output

JPEGEncoderL4TM
^^^^^^^^^^^^^^^

.. code-block:: c++

   JPEGEncoderL4TMProps();
   unsigned short quality; // range 1-100 JPEG Compression quality - 100 means best quality but more file size

   JPEGEncoderL4TM(JPEGEncoderL4TMProps props);

- Currently only supports single channel input

VimbaSource
^^^^^^^^^^^
.. code-block:: c++

   VimbaSourceProps(size_t _maxConcurrentFrames); // 2*fps of the camera is a good value
   // _maxConcurrentFrames is used to announceFrames to the vimbaDriver Camera

   VimbaSource(VimbaSourceProps _props);

EdgeDefectAnalysis
^^^^^^^^^^^^^^^^^^
Sends EdgeDefectAnalysisResult to the next module

.. code-block:: c++

   class EdgeDefectAnalysisConfig
   {
   public:
      std::vector<int> roi;
      std::string edgePos; // "R" "L"

      int gaussianKernelSize;  
      int edgeDefectWidth; 

      // parameters for edge localization
      // https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=canny#canny
      int lowThresh; 
      int highThresh; 
      int phaseLowThresh; 
      int phaseHighThresh;
      //https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=houghlinesp#houghlinesp
      int lineThreshold;
      int minLineLength;
      int maxLineGap;

      // parameters for defect detection
      // https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=canny#canny
      int lowThresh2;
      int highThresh2;
      int phaseLowThresh2;
      int phaseHighThresh2;
      //https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=houghlinesp#houghlinesp
      int lineThreshold2;
      int minLineLength2;
      int maxLineGap2;

      bool enableDebug; // saves the intermediate results 
   }

   class EdgeDefectAnalysisProps: ModuleProps
   {
   public:
      EdgeDefectAnalysisConfig config;
   }

   EdgeDefectAnalysis(EdgeDefectAnalysisProps props);

   class EdgeDefectAnalysisResult
   {
   public:
      	bool isEdgeFound;
         bool isDefect;
         uint64_t fIndex;
         uint64_t timestamp;
   }

LoggerProps
^^^^^^^^^^^

.. code-block:: c++

   bool enableConsoleLog;
   bool enableFileLog;
   std::string fileLogPath;
   boost::log::trivial::severity_level logLevel;

Logger
^^^^^^

.. code-block:: c++

   static void initLogger(LoggerProps props); // valid only once - if called second time - nothing happens. Use the set methods for changes during runtime
   void setLogLevel(boost::log::trivial::severity_level severity); 
   void setConsoleLog(bool enableLog);
   void setFileLog(bool enableLog);

   The below macros are available for logging   
   LOG_TRACE << "A trace severity message";
   LOG_DEBUG << "A debug severity message";
   LOG_INFO << "An informational severity message";
   LOG_WARNING << "A warning severity message";
   LOG_ERROR << "An error severity message";
   LOG_FATAL << "A fatal severity message";

