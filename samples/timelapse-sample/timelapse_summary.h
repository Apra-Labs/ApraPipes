
#include <PipeLine.h>

class TimelapsePipeline {
public:
  TimelapsePipeline();
  bool setupPipeline();
  bool startPipeline();
  bool stopPipeline();
  void test();

private:
  PipeLine pipeline;
};