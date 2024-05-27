#include <PipeLine.h>
#include <ValveModule.h>
#include "Mp4ReaderSource.h"
#include "H264Decoder.h"
#include "ColorConversionXForm.h"
#include "CudaMemCopy.h"
#include "FileWriterModule.h"
#include "JPEGEncoderNVJPEG.h"



class GenerateThumbnailsPipeline
{
public:
    GenerateThumbnailsPipeline(); 
    bool setUpPipeLine();
    bool startPipeLine();
    bool stopPipeLine();
    bool testPipeLine();

private:
    PipeLine pipeLine;
    boost::shared_ptr<ValveModule> valve;
    boost::shared_ptr<Mp4ReaderSource> mp4Reader;
    boost::shared_ptr<H264Decoder> decoder;
    boost::shared_ptr<ColorConversion> colorchange;
    boost::shared_ptr<CudaMemCopy> cudaCopy;
    boost::shared_ptr<JPEGEncoderNVJPEG> jpegEncoder;
    boost::shared_ptr<FileWriterModule> fileWriter;
};

