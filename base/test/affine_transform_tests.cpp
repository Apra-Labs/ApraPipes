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

BOOST_AUTO_TEST_SUITE(affinetransform_test)

BOOST_AUTO_TEST_CASE(cam_mask_rgba)
{
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(NvV4L2CameraProps(400, 400, 2))); /// DMA

	auto nv_transform = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::YUV444))); // DMA
	source->setNext(nv_transform);

	auto stream = cudastream_sp(new ApraCudaStream);
	AffineTransformProps affineProps(AffineTransformProps::CUBIC, stream, 10, 45, 0, 1.0f);
	affineProps.qlen = 1;
	affineProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto affine = boost::shared_ptr<AffineTransform>(new AffineTransform(affineProps));
	nv_transform->setNext(affine);

	auto sync = boost::shared_ptr<Module>(new CudaStreamSynchronize(CudaStreamSynchronizeProps(stream)));
	affine->setNext(sync);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0)));
	sync->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);
	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(100));
	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(rotate_mask)
{
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(NvV4L2CameraProps(640, 480, 2))); /// DMA

	auto nv_transform = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::RGBA))); // DMA
	source->setNext(nv_transform);

	auto stream = cudastream_sp(new ApraCudaStream);
	AffineTransformProps affineProps(AffineTransformProps::CUBIC, stream, 20, 0, 0, 1.0f);
	affineProps.qlen = 1;
	affineProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto affine = boost::shared_ptr<AffineTransform>(new AffineTransform(affineProps));
	nv_transform->setNext(affine);

	auto nv_transform1 = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::YUV444))); // DMA
	affine->setNext(nv_transform1);

	MaskNPPIProps maskProps(320, 240, 100, stream);
	auto m_Mask = boost::shared_ptr<MaskNPPI>(new MaskNPPI(maskProps));
	nv_transform1->setNext(m_Mask);

	auto sync = boost::shared_ptr<Module>(new CudaStreamSynchronize(CudaStreamSynchronizeProps(stream)));
	m_Mask->setNext(sync);

	auto nv_transform2 = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::RGBA))); // DMA
	sync->setNext(nv_transform2);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0)));
	nv_transform2->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::debug);

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(10));
	auto affinecurrprops = affine->getProps();
	affinecurrprops.angle = 10;
	affine->setProps(affinecurrprops);

	boost::this_thread::sleep_for(boost::chrono::seconds(4));
	auto currmaskProps = m_Mask->getProps();
	currmaskProps.maskSelected = MaskNPPIProps::AVAILABLE_MASKS::OCTAGONAL;
	m_Mask->setProps(currmaskProps);
	boost::this_thread::sleep_for(boost::chrono::seconds(4));

	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(cam_mask_rgba_rotate)
{
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(NvV4L2CameraProps(400, 400, 2))); /// DMA

	auto nv_transform = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::RGBA))); // DMA
	source->setNext(nv_transform);

	auto stream = cudastream_sp(new ApraCudaStream);

	AffineTransformProps affineProps(AffineTransformProps::NN, stream, 20, 0, 0, 1.0f);
	auto affine = boost::shared_ptr<AffineTransform>(new AffineTransform(affineProps));
	nv_transform->setNext(affine);

	auto sync = boost::shared_ptr<Module>(new CudaStreamSynchronize(CudaStreamSynchronizeProps(stream)));
	affine->setNext(sync);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0)));
	sync->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::debug);

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(10));
	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(cam_mask_yuv444)
{
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(NvV4L2CameraProps(640, 480, 2))); /// DMA

	auto nv_transform = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::YUV444))); // DMA
	source->setNext(nv_transform);

	auto stream = cudastream_sp(new ApraCudaStream);

	AffineTransformProps affineProps(AffineTransformProps::CUBIC, stream, 0, 0, 0, 1.0f);
	auto affine = boost::shared_ptr<AffineTransform>(new AffineTransform(affineProps));
	nv_transform->setNext(affine);

	auto sync = boost::shared_ptr<Module>(new CudaStreamSynchronize(CudaStreamSynchronizeProps(stream)));
	affine->setNext(sync);

	auto dmaToHostCopy = boost::shared_ptr<Module>(new DMAFDToHostCopy);
	sync->setNext(dmaToHostCopy);

	auto fileWriter = boost::shared_ptr<FileWriterModule>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/YUV640/Frame.raw")));
	dmaToHostCopy->setNext(fileWriter);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0)));
	sync->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());
	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(5));

	Logger::setLogLevel(boost::log::trivial::severity_level::debug);
	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(cam_mask_yuv444_rotate)
{
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(NvV4L2CameraProps(400, 400, 2))); /// DMA

	auto nv_transform = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::YUV444))); // DMA
	source->setNext(nv_transform);

	auto stream = cudastream_sp(new ApraCudaStream);

	AffineTransformProps affineProps(AffineTransformProps::CUBIC, stream, 15, 0, 0, 1.0f);
	auto affine = boost::shared_ptr<AffineTransform>(new AffineTransform(affineProps));
	nv_transform->setNext(affine);

	auto sync = boost::shared_ptr<Module>(new CudaStreamSynchronize(CudaStreamSynchronizeProps(stream)));
	affine->setNext(sync);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0)));
	sync->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());
	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(10));
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);
	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(cam_mask_yuv420_rotate)
{
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(NvV4L2CameraProps(400, 400, 2))); /// DMA

	auto nv_transform = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::YUV420))); // DMA
	source->setNext(nv_transform);

	auto stream = cudastream_sp(new ApraCudaStream);

	AffineTransformProps affineProps(AffineTransformProps::CUBIC, stream, 15, 0, 0, 1.0f);
	auto affine = boost::shared_ptr<AffineTransform>(new AffineTransform(affineProps));
	nv_transform->setNext(affine);

	auto sync = boost::shared_ptr<Module>(new CudaStreamSynchronize(CudaStreamSynchronizeProps(stream)));
	affine->setNext(sync);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0)));
	sync->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());
	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(10));
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);
	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(transformyuv444)
{
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(NvV4L2CameraProps(400, 400, 2))); /// DMA

	auto nv_transform = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::YUV444))); // DMA
	source->setNext(nv_transform);

	auto dmaToHostCopy = boost::shared_ptr<Module>(new DMAFDToHostCopy);
	nv_transform->setNext(dmaToHostCopy);

	auto fileWriter = boost::shared_ptr<FileWriterModule>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/YUV444/Frame.raw")));
	dmaToHostCopy->setNext(fileWriter);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());
	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(10));
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);
	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_SUITE_END()
