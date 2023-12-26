#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include "FileWriterModule.h"
#include "FileReaderModule.h"
#include "ExternalSinkModule.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "AIPExceptions.h"
#include "CudaMemCopy.h"
#include "CudaStreamSynchronize.h"
#include "MaskNPPI.h"
#include "NvV4L2Camera.h"
#include "EglRenderer.h"
#include "NvTransform.h"
#include "PipeLine.h"
#include "DMAFDToHostCopy.h"
#include <chrono>
#include "AffineTransform.h"

using sys_clock = std::chrono::system_clock;

#include "test_utils.h"

BOOST_AUTO_TEST_SUITE(at_tests)

BOOST_AUTO_TEST_CASE(cam)
{
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(NvV4L2CameraProps(640, 480, 2))); /// DMA
	// source will get YUYV
	auto nv_transform = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::YUV444))); //DMA
	source->setNext(nv_transform);
	// transforming YUYV to YUV444
	auto stream = cudastream_sp(new ApraCudaStream);

	
    AffineTransformProps affineProps(AffineTransformProps::NN, stream, 0, 0, 0, 1.0f);
    auto affine = boost::shared_ptr<AffineTransform>(new AffineTransform(affineProps));
    nv_transform->setNext(affine);

	auto sync = boost::shared_ptr<Module>(new CudaStreamSynchronize(CudaStreamSynchronizeProps(stream)));
	affine->setNext(sync);

	// auto nv_transform2 = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::RGBA)));
	// sync->setNext(nv_transform2);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0)));
	sync->setNext(sink);

	//auto dmaToHostCopy = boost::shared_ptr<Module>(new DMAFDToHostCopy);
	//sync->setNext(dmaToHostCopy);

	//auto fileWriter = boost::shared_ptr<FileWriterModule>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/MaskOutputImages/Frame???.raw")));
	//dmaToHostCopy->setNext(fileWriter);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(100));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);
	p.stop();
	p.term();
	p.wait_for_all();
}
BOOST_AUTO_TEST_SUITE_END()
