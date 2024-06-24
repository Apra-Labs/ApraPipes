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
    bool setUpPipeLine(const std::string &videoPath,const std::string &outFolderPath);
    bool startPipeLine();
    bool stopPipeLine();

private:
    PipeLine pipeLine;
    boost::shared_ptr<ValveModule> mValve;
    boost::shared_ptr<Mp4ReaderSource> mMp4Reader;
    boost::shared_ptr<H264Decoder> mDecoder;
    boost::shared_ptr<ColorConversion> mColorchange;
    boost::shared_ptr<CudaMemCopy> mCudaCopy;
    boost::shared_ptr<JPEGEncoderNVJPEG> mJpegEncoder;
    boost::shared_ptr<FileWriterModule> mFileWriter;
};

