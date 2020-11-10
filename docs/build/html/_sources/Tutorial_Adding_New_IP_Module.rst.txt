Adding New Module
=================
ChangeDetection Module 

- Input is histogram bins
- Output is ChangeDetectionResult


Define Module Properties
""""""""""""""""""""""""
.. code-block:: c++

   class ChangeDetectionProps: public ModuleProps
   {
   public:
      ChangeDetectionProps(): ModuleProps()
      {
         refWindowLength = 1;
         refDelayLength = -1;
         insWindowLength = 1;
         threshold = 1;
         compareMethod = 1;
      }

      ChangeDetectionProps(int _refWindowLength, int _refDelayLength, int _insWindowLength, double _threshold, int _compareMethod): ModuleProps()
      {
         refWindowLength = _refWindowLength;
         refDelayLength = _refDelayLength;
         insWindowLength = _insWindowLength;
         threshold = _threshold;
         compareMethod = _compareMethod;	
      }

      // All the properties can be updated during run time using setProps
      int refWindowLength;
      int refDelayLength;
      int insWindowLength;
      double threshold;
      int compareMethod;	

   private:
      friend class boost::serialization::access;
      
      template<class Archive>
      void serialize(Archive &ar, const unsigned int version)
      {
         ar & boost::serialization::base_object<ModuleProps>(*this);
         ar & refWindowLength;
         ar & refDelayLength;
         ar & insWindowLength;
         ar & threshold;
         ar & compareMethod;
      }
   };

Validating the input and output
"""""""""""""""""""""""""""""""
.. code-block:: c++


   bool ChangeDetection::validateInputOutputPins()
   {
      // one and only 1 array should exist
      auto count = getNumberOfInputsByType(FrameMetadata::ARRAY);
      if (count != 1)
      {
         LOG_ERROR << "Input pin of type ARRAY is expected.";
         return false;
      }

      // output CHANGE_DETECTION pin should exist
      count = getNumberOfOutputsByType(FrameMetadata::CHANGE_DETECTION);
      if (count != 1)
      {
         LOG_ERROR << "Input pin of type CHANGE_DETECTION is expected.";
         return false;
      }

      return true;
   }

Initialization
""""""""""""""
.. code-block:: c++

   bool ChangeDetection::init()
   {
      if (!Module::init())
      {
         return false;
      }

      // any initialization here     

      return true;
   }

Handling the first frame and using the input metadata
"""""""""""""""""""""""""""""""""""""""""""""""""""""
.. code-block:: c++

   bool ChangeDetection::processSOS(frame_sp& frame)
   {
      auto metadata = frame->getMetadata();
      if (metadata->getFrameType() != FrameMetadata::ARRAY)
      {
         return true;
      }

      // metadata has width, height, type depending on the frame type
      
      return true;
   }


Output
""""""
.. code-block:: c++

   class ChangeDetectionResult
   {
   public:

      ChangeDetectionResult(bool changeDetected, double distance, uint64_t index)
      {
         mChangeDetected = changeDetected;
         mDistance = distance;
         fIndex = index;
      }

      ChangeDetectionResult() {}

      static boost::shared_ptr<ChangeDetectionResult> deSerialize(frame_container& frames)
      {
         auto frameType = FrameMetadata::CHANGE_DETECTION; 

         auto frame = frame_sp();  
         for (auto it = frames.cbegin(); it != frames.cend(); it++)
         {
            auto tempFrame = it->second;
            if (tempFrame->getMetadata()->getFrameType() == frameType)
            {
               frame = tempFrame;
            }
         }

         if (!frame.get())
         {
            return boost::shared_ptr<ChangeDetectionResult>();
         }

         auto result = boost::shared_ptr<ChangeDetectionResult>(new ChangeDetectionResult(false, 0, 0));
         auto& obj = *result.get();
         Utils::deSerialize<ChangeDetectionResult>(obj, frame->data(), frame->size());

         return result;	
      }

      static void serialize(bool changeDetected, double distance, uint64_t index, void* buffer, size_t size)
      {
         auto result = ChangeDetectionResult(changeDetected, distance, index);
         Utils::serialize<ChangeDetectionResult>(result, buffer, size); 
      }			

      static size_t getSerializeSize()
      {
         return 1024 + sizeof(mChangeDetected) + sizeof(mDistance) + sizeof(fIndex);
      }

      bool mChangeDetected;
      double mDistance;
      uint64_t fIndex;

   private:
      friend class boost::serialization::access;
      template<class Archive>
      void serialize(Archive & ar, const unsigned int /* file_version */) {
         ar & mChangeDetected & mDistance & fIndex;
      }
   };

Consuming the input and send output
"""""""""""""""""""""""""""""""""""
.. code-block:: c++

   bool ChangeDetection::process(frame_container& frames)
   {
      auto inFrame = getFrameByType(frames, FrameMetadata::ARRAY);
      auto metadata = mDetail->getOutputMetadata();
      auto outFrame = makeFrame(ChangeDetectionResult::getSerializeSize(), metadata);

      // do the computation here

      auto pinId = getOutputPinIdByType(FrameMetadata::CHANGE_DETECTION);
      frames.insert(make_pair(pinId, outFrame));
      send(frames);

      return true;
   }
