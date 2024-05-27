#include <PipeLine.h>
#include "Mp4ReaderSource.h"
#include "ImageViewerModule.h"
#include "ColorConversionXForm.h"
#include "H264Decoder.h"


class PlayMp4VideoFromBeginning {
public:
    PlayMp4VideoFromBeginning(); 
    bool setUpPipeLine();
    bool startPipeLine();
    bool stopPipeLine();
    bool flushQueuesAndSeek();
    bool testPipeLineForFlushQue();
    bool testPipeLineForSeek();

private:
    PipeLine pipeLine;
    boost::shared_ptr<Mp4ReaderSource> mp4Reader;
    boost::shared_ptr<ImageViewerModule> imageViewerSink;
    boost::shared_ptr<H264Decoder> decoder;
    boost::shared_ptr<ColorConversion> colorchange;
};