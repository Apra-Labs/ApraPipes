#include <PipeLine.h>
#include <ValveModule.h>

class GenerateThumbnailsPipeline
{
public:
    GenerateThumbnailsPipeline(); 
    bool setUpPipeLine();
    bool startPipeLine();
    bool stopPipeLine();

private:
    PipeLine pipeLine;
    boost::shared_ptr<ValveModule> valve;
};