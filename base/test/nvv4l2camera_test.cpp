#include <boost/test/unit_test.hpp>
#include <unistd.h> // Include the <unistd.h> header for usleep

#include "PipeLine.h"
#include "NvV4L2Camera.h"
#include "NvTransform.h"
#include "VirtualCameraSink.h"
#include "FileWriterModule.h"
#include "DMAFDToHostCopy.h"
#include "StatSink.h"
#include "EglRenderer.h"
#include "ThumbnailListGenerator.h"
#include "ImageEncoderCV.h"
#include "ColorConversionXForm.h"
#include "AffineTransform.h"
#include "MaskNPPI.h"
#include "CudaStreamSynchronize.h"
#include "Mp4ReaderSource.h"
#include "SquareMaskNPPI.h"

BOOST_AUTO_TEST_SUITE(nvv4l2camera_tests)

BOOST_AUTO_TEST_CASE(basic, *boost::unit_test::disabled())
{
	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(NvV4L2CameraProps(640, 480, 10)));

	StatSinkProps sinkProps;
	sinkProps.logHealth = true;
	sinkProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new StatSink(sinkProps));
	source->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(10));
	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(render_2, *boost::unit_test::disabled())
{
	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(NvV4L2CameraProps(400, 400, 2)));


	NvTransformProps nvProps(ImageMetadata::RGBA);
	nvProps.qlen = 1;
	auto nv_transform = boost::shared_ptr<Module>(new NvTransform(nvProps));
	source->setNext(nv_transform);

	EglRendererProps eglProps(0,0);
	eglProps.qlen = 1;
	auto sink = boost::shared_ptr<Module>(new EglRenderer(eglProps));
	nv_transform->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(100));
	p.stop();
	p.term();
	p.wait_for_all();
}


BOOST_AUTO_TEST_CASE(save, *boost::unit_test::disabled())
{
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(NvV4L2CameraProps(400, 400, 5)));

	auto nv_transform = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::RGBA)));
	source->setNext(nv_transform);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0)));
	nv_transform->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(100000));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(thumbnail, *boost::unit_test::disabled())
{
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(NvV4L2CameraProps(640, 480, 10)));

	auto nv_transform = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::RGBA)));
	source->setNext(nv_transform);

	auto copySource = boost::shared_ptr<Module>(new DMAFDToHostCopy);
	nv_transform->setNext(copySource);

	// auto conversionType = ColorConversionProps::ConversionType::YUV420PLANAR_TO_RGB;
	// auto colorchange = boost::shared_ptr<ColorConversion>(new ColorConversion(ColorConversionProps(conversionType)));
	// copySource->setNext(colorchange);

	// auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/rgb/frame_????.raw", false)));
	// colorchange->setNext(fileWriter);

	// auto thumbnailgenerator = boost::shared_ptr<Module>(new ImageEncoderCV(ImageEncoderCVProps()));
	// copySource->setNext(thumbnailgenerator);

	auto thumbnailgenerator = boost::shared_ptr<ThumbnailListGenerator>(new ThumbnailListGenerator(ThumbnailListGeneratorProps(200, 200, "/home/developer/workspace/ApraPipes/test_1212.jpeg")));
	copySource->setNext(thumbnailgenerator);

	// auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/th/frame_????.png")));
	// copySource->setNext(fileWriter);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(4));

	// auto currThumbnailProps = thumbnailgenerator->getProps();
	// currThumbnailProps.fileToStore = "/home/developer/workspace/ApraPipes/test13.jpeg";
	// thumbnailgenerator->setProps(currThumbnailProps);
	boost::this_thread::sleep_for(boost::chrono::seconds(4));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(vcam, *boost::unit_test::disabled())
{
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(NvV4L2CameraProps(1280, 720, 10)));

	auto copySource = boost::shared_ptr<Module>(new DMAFDToHostCopy);
	source->setNext(copySource);

	auto transform = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::YUV420)));
	source->setNext(transform);

	auto copy = boost::shared_ptr<Module>(new DMAFDToHostCopy);
	transform->setNext(copy);

	auto transform2 = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::NV12)));
	source->setNext(transform2);

	auto copy2 = boost::shared_ptr<Module>(new DMAFDToHostCopy);
	transform2->setNext(copy2);

	// VirtualCameraSinkProps sinkProps("/dev/video10");
	// sinkProps.logHealth = true;
	// sinkProps.logHealthFrequency = 100;
	// auto sink = boost::shared_ptr<Module>(new VirtualCameraSink(sinkProps));
	// copy->setNext(sink);

	auto fileWriter1 = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/nvv4l2/uyvy_????.raw")));
	copySource->setNext(fileWriter1);

	auto fileWriter2 = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/nvv4l2/yuv420_????.raw")));
	copy->setNext(fileWriter2);

	// auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0)));
	// transform->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(10));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(atlcam, *boost::unit_test::disabled()) // Getting latency of 130ms, previously we have got around range og 60 to 130
{
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);
	NvV4L2CameraProps nvCamProps(640, 480, 2);
	nvCamProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;

	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(nvCamProps));

	// auto copySource = boost::shared_ptr<Module>(new DMAFDToHostCopy);
	// source->setNext(copySource);

	// auto fileWriter1 = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/nvv4l2/Frame_????.raw")));
	// copySource->setNext(fileWriter1);

	NvTransformProps nvprops(ImageMetadata::YUV444);
	nvprops.qlen = 1;
	nvprops.logHealth = true;
	nvprops.logHealthFrequency = 100;
	nvprops.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto transform = boost::shared_ptr<Module>(new NvTransform(nvprops));
	source->setNext(transform);

	// auto copySource1 = boost::shared_ptr<Module>(new DMAFDToHostCopy);
	// transform->setNext(copySource1);

	// auto fileWriter11 = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/nvv4l2232/Frame2_????.raw")));
	// copySource1->setNext(fileWriter11);

	EglRendererProps eglProps(0, 0);
	eglProps.qlen = 1;
	eglProps.fps = 60;
	eglProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	eglProps.logHealth = true;
	eglProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new EglRenderer(eglProps));
	transform->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::debug);

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(1000000));
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(test_new_atl, *boost::unit_test::disabled()) // Getting latency of 130ms, previously we have got around range og 60 to 130
{
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);
	NvV4L2CameraProps nvCamProps(720, 720, 3);
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(nvCamProps));

	NvTransformProps nvprops(ImageMetadata::RGBA);
	nvprops.qlen = 1;
	nvprops.logHealth = true;
	nvprops.logHealthFrequency = 100;
	nvprops.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;

	auto transform = boost::shared_ptr<Module>(new NvTransform(nvprops));
	source->setNext(transform);

	EglRendererProps eglProps(0, 0);
	eglProps.qlen = 1;
	eglProps.fps = 60;

	auto sink = boost::shared_ptr<Module>(new EglRenderer(eglProps));
	transform->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::debug);

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(1000));
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(camLatency420, *boost::unit_test::disabled())
{
	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	// 1. Source 400x400
	NvV4L2CameraProps sourceProps(400, 400, 2);
	sourceProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;

	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(sourceProps));

	// 2. Take Input from Source (400 x 400) and apply Nv Transform to scale to 1000 x 1000
	NvTransformProps nvProps(ImageMetadata::RGBA, 1000, 1000);
	nvProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	nvProps.qlen =1;
	auto nv_transform = boost::shared_ptr<NvTransform>(new NvTransform(nvProps));
	source->setNext(nv_transform);

	// 3. EGL Renderer to display real time stream
	EglRendererProps eglProps(0, 0);
	eglProps.qlen=1;
	auto sink = boost::shared_ptr<Module>(new EglRenderer(eglProps));
	nv_transform->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(1000));
	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(camLatency720, *boost::unit_test::disabled())
{
	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	// 1. Source 720x720
	NvV4L2CameraProps sourceProps(720, 720, 2);
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(sourceProps));

	// 2. Take Input from Source (720 x 720) and apply Nv Transform to scale to 1000 x 1000
	NvTransformProps nvProps(ImageMetadata::RGBA, 1000, 1000);
	nvProps.qlen = 1;
	auto nv_transform = boost::shared_ptr<NvTransform>(new NvTransform(nvProps));
	source->setNext(nv_transform);

	// 3. EGL Renderer to display real time stream
	EglRendererProps eglProps(0, 0);
	eglProps.qlen = 1;
	auto sink = boost::shared_ptr<Module>(new EglRenderer(eglProps));
	nv_transform->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(1000));
	p.stop();
	p.term();
	p.wait_for_all();
}
// BOOST_AUTO_TEST_CASE(atlcamwithaffine, *boost::unit_test::disabled())
// {
// 	Logger::setLogLevel(boost::log::trivial::severity_level::debug);
// 	NvV4L2CameraProps nvCamProps(400, 400, 2);
// 	nvCamProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
// 	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(nvCamProps));

// 	NvTransformProps nvprops(ImageMetadata::YUV444);
// 	nvprops.qlen = 1;
// 	nvprops.logHealth = true;
// 	nvprops.logHealthFrequency = 100;
// 	nvprops.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
// 	auto transform = boost::shared_ptr<Module>(new NvTransform(nvprops));
// 	source->setNext(transform);

// 	auto stream = cudastream_sp(new ApraCudaStream);

// 	AffineTransformProps affineProps(AffineTransformProps::CUBIC, stream, 0, 0, 0, 2);
// 	affineProps.qlen = 1;
// 	affineProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
// 	auto m_affineTransform = boost::shared_ptr<AffineTransform>(new AffineTransform(affineProps));
// 	transform->setNext(m_affineTransform);

// 	NvTransformProps nvprops1(ImageMetadata::RGBA);
// 	nvprops1.qlen = 2;
// 	nvprops1.logHealth = true;
// 	nvprops1.logHealthFrequency = 100;
// 	nvprops1.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
// 	auto m_rgbaTransform = boost::shared_ptr<NvTransform>(new NvTransform(nvprops));
// 	m_affineTransform->setNext(m_rgbaTransform);

// 	// auto sync = boost::shared_ptr<CudaStreamSynchronize>(new CudaStreamSynchronize(CudaStreamSynchronizeProps(stream)));
// 	// m_affineTransform->setNext(sync);

// 	EglRendererProps eglProps(0, 0);
// 	eglProps.qlen = 1;
// 	eglProps.fps = 60;
// 	eglProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
// 	eglProps.logHealth = true;
// 	eglProps.logHealthFrequency = 100;
// 	auto sink = boost::shared_ptr<Module>(new EglRenderer(eglProps));
// 	m_rgbaTransform->setNext(sink);

// 	PipeLine p("test");
// 	p.appendModule(source);
// 	BOOST_TEST(p.init());

// 	Logger::setLogLevel(boost::log::trivial::severity_level::debug);

// 	p.run_all_threaded();
// 	boost::this_thread::sleep_for(boost::chrono::seconds(1000000));
// 	Logger::setLogLevel(boost::log::trivial::severity_level::debug);

// 	p.stop();
// 	p.term();

// 	p.wait_for_all();
// }

BOOST_AUTO_TEST_CASE(atlcamwithaffine, *boost::unit_test::disabled())
{
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);
	NvV4L2CameraProps nvCamProps(400, 400, 2);
	nvCamProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(nvCamProps));

	NvTransformProps nvprops(ImageMetadata::RGBA);
	nvprops.qlen = 1;
	nvprops.logHealth = true;
	nvprops.logHealthFrequency = 100;
	nvprops.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto transform = boost::shared_ptr<Module>(new NvTransform(nvprops));
	source->setNext(transform);

	// auto stream = cudastream_sp(new ApraCudaStream);

	// AffineTransformProps affineProps(AffineTransformProps::CUBIC, stream, 0,1792, 0, 0, 1);
	// affineProps.qlen = 1;
	// affineProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	// auto m_affineTransform = boost::shared_ptr<AffineTransform>(new AffineTransform(affineProps));
	// transform->setNext(m_affineTransform);

	// auto sync = boost::shared_ptr<CudaStreamSynchronize>(new CudaStreamSynchronize(CudaStreamSynchronizeProps(stream)));
	// m_affineTransform->setNext(sync);

	// auto copySource = boost::shared_ptr<DMAFDToHostCopy>(new DMAFDToHostCopy);
	// sync->setNext(copySource);


	// auto fileWriter1 = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/affine/Frame_????.raw")));
	// // sync->setNext(fileWriter1);

	EglRendererProps eglProps(0, 0);
	eglProps.qlen = 1;
	eglProps.fps = 30;
	eglProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	eglProps.logHealth = true;
	eglProps.logHealthFrequency = 100;

	auto sink = boost::shared_ptr<Module>(new EglRenderer(eglProps));
	transform->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::debug);

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(1000000));
	// double angle = 0;
	// while(true)
	// {
	// 	angle+=90;
	// 	if(angle>=359)
	// 	{
	// 		angle = 0;
	// 	}
	// 	auto affineCurrProps = m_affineTransform->getProps();
	// 	affineCurrProps.angle = angle;
	// 	m_affineTransform->setProps(affineCurrProps);
	// 	boost::this_thread::sleep_for(boost::chrono::seconds(1));
	// }
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);
	p.stop();
	p.term();

	p.wait_for_all();
}
BOOST_AUTO_TEST_CASE(atlcamwithaffine1, *boost::unit_test::disabled())
{
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);
	NvV4L2CameraProps nvCamProps(400, 400, 2);
	nvCamProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(nvCamProps));

	NvTransformProps nvprops(ImageMetadata::YUV444);
	nvprops.qlen = 1;
	nvprops.logHealth = true;
	nvprops.logHealthFrequency = 100;
	nvprops.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto transform = boost::shared_ptr<Module>(new NvTransform(nvprops));
	source->setNext(transform);

	auto stream = cudastream_sp(new ApraCudaStream);

	AffineTransformProps affineProps(AffineTransformProps::CUBIC, stream, 0,1792, 0, 0, 1);
	affineProps.qlen = 1;
	affineProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto m_affineTransform = boost::shared_ptr<AffineTransform>(new AffineTransform(affineProps));
	transform->setNext(m_affineTransform);

	// auto sync = boost::shared_ptr<CudaStreamSynchronize>(new CudaStreamSynchronize(CudaStreamSynchronizeProps(stream)));
	// m_affineTransform->setNext(sync);

	EglRendererProps eglProps(0, 0);
	eglProps.qlen = 1;
	eglProps.fps = 30;
	eglProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	eglProps.logHealth = true;
	eglProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new EglRenderer(eglProps));
	m_affineTransform->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::debug);

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(1000000));
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(atlcamwithaffinergba, *boost::unit_test::disabled())
{
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);
	NvV4L2CameraProps nvCamProps(400, 400, 2);
	nvCamProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(nvCamProps));

	NvTransformProps nvprops(ImageMetadata::RGBA);
	nvprops.qlen = 2;
	nvprops.logHealth = true;
	nvprops.logHealthFrequency = 100;
	nvprops.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto transform = boost::shared_ptr<Module>(new NvTransform(nvprops));
	source->setNext(transform);

	auto stream = cudastream_sp(new ApraCudaStream);

	// AffineTransformProps affineProps(AffineTransformProps::CUBIC, stream, 0,4096, 0, 0, 2.5);
	AffineTransformProps affineProps(AffineTransformProps::CUBIC, stream, 0,1792, 0, 0, 1);
	affineProps.qlen = 2;
	affineProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto m_affineTransform = boost::shared_ptr<AffineTransform>(new AffineTransform(affineProps));
	transform->setNext(m_affineTransform);

    CudaStreamSynchronizeProps cuctxprops(stream);
	cuctxprops.qlen = 2;
	cuctxprops.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto sync = boost::shared_ptr<CudaStreamSynchronize>(new CudaStreamSynchronize(cuctxprops));
	m_affineTransform->setNext(sync);

	EglRendererProps eglProps(0 ,0, 400, 400);
	eglProps.qlen = 2;
	eglProps.fps = 20;
	eglProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	eglProps.logHealth = true;
	eglProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<EglRenderer>(new EglRenderer(eglProps));
	m_affineTransform->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::debug);

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(5));
	auto currAffineProps = m_affineTransform->getProps();
	currAffineProps.angle = 90.0;
	m_affineTransform->setProps(currAffineProps);
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);
	boost::this_thread::sleep_for(boost::chrono::seconds(50));

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(latency, *boost::unit_test::disabled())
{
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);
	NvV4L2CameraProps nvCamProps(400, 400, 1);
	nvCamProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(nvCamProps));

	NvTransformProps nvprops(ImageMetadata::RGBA);
	nvprops.qlen = 2;
	// nvprops.logHealth = true;
	// nvprops.logHealthFrequency = 100;
	nvprops.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto transform = boost::shared_ptr<Module>(new NvTransform(nvprops));
	source->setNext(transform);

	EglRendererProps eglProps(0 ,0, 400, 400);
	eglProps.qlen = 2;
	eglProps.fps = 60;
	eglProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	// eglProps.logHealth = true;
	// eglProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<EglRenderer>(new EglRenderer(eglProps));
	transform->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::debug);

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(5000000));

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(atlcamwithaffinewithsync, *boost::unit_test::disabled())
{
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);
	NvV4L2CameraProps nvCamProps(400, 400, 2);
	nvCamProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(nvCamProps));

	NvTransformProps nvprops(ImageMetadata::YUV444);
	nvprops.qlen = 1;
	nvprops.logHealth = true;
	nvprops.logHealthFrequency = 100;
	nvprops.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto transform = boost::shared_ptr<Module>(new NvTransform(nvprops));
	source->setNext(transform);

	auto stream = cudastream_sp(new ApraCudaStream);

	AffineTransformProps affineProps(AffineTransformProps::CUBIC, stream, 0,1792, 0, 0, 1);
	affineProps.qlen = 1;
	affineProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto m_affineTransform = boost::shared_ptr<AffineTransform>(new AffineTransform(affineProps));
	transform->setNext(m_affineTransform);

	NvTransformProps nvprops1(ImageMetadata::RGBA);
	nvprops1.qlen = 2;
	nvprops1.logHealth = true;
	nvprops1.logHealthFrequency = 100;
	nvprops1.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto m_rgbaTransform = boost::shared_ptr<NvTransform>(new NvTransform(nvprops));
	m_affineTransform->setNext(m_rgbaTransform);

	auto sync = boost::shared_ptr<CudaStreamSynchronize>(new CudaStreamSynchronize(CudaStreamSynchronizeProps(stream)));
	m_rgbaTransform->setNext(sync);

	EglRendererProps eglProps(0, 0);
	eglProps.qlen = 1;
	eglProps.fps = 30;
	eglProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	eglProps.logHealth = true;
	eglProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new EglRenderer(eglProps));
	sync->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::debug);

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(1000000));
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(atlcamwithmask, *boost::unit_test::disabled())
{
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);
	NvV4L2CameraProps nvCamProps(400, 400, 2);
	nvCamProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(nvCamProps));

	NvTransformProps nvprops(ImageMetadata::RGBA);
	nvprops.qlen = 1;
	nvprops.logHealth = true;
	nvprops.logHealthFrequency = 100;
	nvprops.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto transform = boost::shared_ptr<Module>(new NvTransform(nvprops));
	source->setNext(transform);

	auto stream = cudastream_sp(new ApraCudaStream);

	// MaskNPPIProps maskProps(200, 200, 100, MaskNPPIProps::AVAILABLE_MASKS::CIRCLE, stream);
	// maskProps.qlen = 1;
	// maskProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	// auto m_mask = boost::shared_ptr<MaskNPPI>(new MaskNPPI(maskProps));
	// transform->setNext(m_mask);

	SquareMaskNPPIProps maskProps(300 , stream);
	maskProps.qlen = 1;
	maskProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto m_mask = boost::shared_ptr<SquareMaskNPPI>(new SquareMaskNPPI(maskProps));
	transform->setNext(m_mask);

	CudaStreamSynchronizeProps cuxtxProps(stream);
	cuxtxProps.qlen = 1;
	cuxtxProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto sync = boost::shared_ptr<CudaStreamSynchronize>(new CudaStreamSynchronize(cuxtxProps));
	m_mask->setNext(sync);

	EglRendererProps eglProps(0, 0);
	eglProps.qlen = 1;
	eglProps.fps = 60;
	eglProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	eglProps.logHealth = true;
	eglProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new EglRenderer(eglProps));
	sync->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::debug);

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(1000000));
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(atlcamwithmaskoctagonal, *boost::unit_test::disabled())
{
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);
	NvV4L2CameraProps nvCamProps(400, 400, 2);
	nvCamProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(nvCamProps));

	NvTransformProps nvprops(ImageMetadata::RGBA);
	nvprops.qlen = 1;
	nvprops.logHealth = true;
	nvprops.logHealthFrequency = 100;
	nvprops.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto transform = boost::shared_ptr<Module>(new NvTransform(nvprops));
	source->setNext(transform);

	auto stream = cudastream_sp(new ApraCudaStream);

	MaskNPPIProps maskProps(200, 200, 100, MaskNPPIProps::AVAILABLE_MASKS::OCTAGONAL, stream);
	maskProps.qlen = 1;
	maskProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto m_mask = boost::shared_ptr<MaskNPPI>(new MaskNPPI(maskProps));
	transform->setNext(m_mask);

	CudaStreamSynchronizeProps cuxtxProps(stream);
	cuxtxProps.qlen = 1;
	cuxtxProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto sync = boost::shared_ptr<CudaStreamSynchronize>(new CudaStreamSynchronize(cuxtxProps));
	m_mask->setNext(sync);

	EglRendererProps eglProps(0, 0);
	eglProps.qlen = 1;
	eglProps.fps = 60;
	eglProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	eglProps.logHealth = true;
	eglProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new EglRenderer(eglProps));
	sync->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::debug);

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(1000000));
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(atl_affine_mask, *boost::unit_test::disabled())
{
	// Logger::setLogLevel(boost::log::trivial::severity_level::debug);
	NvV4L2CameraProps nvCamProps(400, 400, 2);
	nvCamProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(nvCamProps));

	NvTransformProps nvprops(ImageMetadata::RGBA);
	nvprops.qlen = 1;
	nvprops.logHealth = true;
	nvprops.logHealthFrequency = 100;
	nvprops.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto transform = boost::shared_ptr<Module>(new NvTransform(nvprops));
	source->setNext(transform);

	auto stream = cudastream_sp(new ApraCudaStream);

	AffineTransformProps affineProps(AffineTransformProps::CUBIC, stream, 0,4096, 0, 0, 2.5);
	affineProps.qlen = 2;
	affineProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto m_affineTransform = boost::shared_ptr<AffineTransform>(new AffineTransform(affineProps));
	transform->setNext(m_affineTransform);

	MaskNPPIProps maskProps(500, 500, 500, MaskNPPIProps::AVAILABLE_MASKS::CIRCLE, stream);
	maskProps.qlen = 1;
	maskProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto m_mask = boost::shared_ptr<MaskNPPI>(new MaskNPPI(maskProps));
	m_affineTransform->setNext(m_mask);

	CudaStreamSynchronizeProps cuxtxProps(stream);
	cuxtxProps.qlen = 1;
	cuxtxProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto sync = boost::shared_ptr<CudaStreamSynchronize>(new CudaStreamSynchronize(cuxtxProps));
	m_mask->setNext(sync);

	auto copySource = boost::shared_ptr<Module>(new DMAFDToHostCopy);
	sync->setNext(copySource);

	auto thumbnailgenerator = boost::shared_ptr<ThumbnailListGenerator>(new ThumbnailListGenerator(ThumbnailListGeneratorProps(200, 200, "/home/developer/workspace/ApraPipes/test_12122.jpeg")));
	copySource->setNext(thumbnailgenerator);

	EglRendererProps eglProps(400, 50, 1000, 1000);
	eglProps.qlen = 1;
	eglProps.fps = 60;
	eglProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	eglProps.logHealth = true;
	eglProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new EglRenderer(eglProps));
	sync->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	// Logger::setLogLevel(boost::log::trivial::severity_level::debug);

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(4));
	// Logger::setLogLevel(boost::log::trivial::severity_level::debug);

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(camScale720To1000, *boost::unit_test::disabled())
{
	// Logger::setLogLevel(boost::log::trivial::severity_level::debug);
	auto stream = cudastream_sp(new ApraCudaStream);
	NvV4L2CameraProps nvCamProps(720, 720, 2);
	nvCamProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(nvCamProps));

	NvTransformProps nvprops(ImageMetadata::RGBA);
	nvprops.qlen = 1;
	nvprops.logHealth = true;
	nvprops.logHealthFrequency = 100;
	nvprops.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto transform = boost::shared_ptr<Module>(new NvTransform(nvprops));
	source->setNext(transform);

	AffineTransformProps affineProps(AffineTransformProps::CUBIC, stream, 0, 4096, 0, 0, 1.4);
	affineProps.qlen = 1;
	affineProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto m_affineTransform = boost::shared_ptr<AffineTransform>(new AffineTransform(affineProps));
	transform->setNext(m_affineTransform);

	CudaStreamSynchronizeProps cuxtxProps(stream);
	cuxtxProps.qlen = 1;
	cuxtxProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto sync = boost::shared_ptr<CudaStreamSynchronize>(new CudaStreamSynchronize(cuxtxProps));
	m_affineTransform->setNext(sync);

	EglRendererProps eglProps(400, 50, 1000, 1000);
	eglProps.qlen = 1;
	eglProps.fps = 60;
	eglProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	eglProps.logHealth = true;
	eglProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new EglRenderer(eglProps));
	sync->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());
	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(1000));
	// Logger::setLogLevel(boost::log::trivial::severity_level::debug);

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(atlcamwithaffinewithmask, *boost::unit_test::disabled())
{
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);

	NvV4L2CameraProps nvCamProps(400, 400, 2);
	nvCamProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(nvCamProps));

	NvTransformProps nvprops(ImageMetadata::YUV444);
	nvprops.qlen = 1;
	nvprops.logHealth = true;
	nvprops.logHealthFrequency = 100;
	nvprops.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto transform = boost::shared_ptr<Module>(new NvTransform(nvprops));
	source->setNext(transform);

	auto stream = cudastream_sp(new ApraCudaStream);
	AffineTransformProps affineProps(AffineTransformProps::CUBIC, stream, 0,1792, 0, 0, 1);
	affineProps.qlen = 1;
	affineProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto m_affineTransform = boost::shared_ptr<AffineTransform>(new AffineTransform(affineProps));
	transform->setNext(m_affineTransform);

	MaskNPPIProps maskProps(200, 200, 100, MaskNPPIProps::AVAILABLE_MASKS::NONE, stream);
	maskProps.qlen = 1;
	maskProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto m_mask = boost::shared_ptr<MaskNPPI>(new MaskNPPI(maskProps));
	m_affineTransform->setNext(m_mask);

	auto sync = boost::shared_ptr<CudaStreamSynchronize>(new CudaStreamSynchronize(CudaStreamSynchronizeProps(stream)));
	m_mask->setNext(sync);

	EglRendererProps eglProps(0, 0);
	eglProps.qlen = 1;
	eglProps.fps = 60;
	eglProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	eglProps.logHealth = true;
	eglProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new EglRenderer(eglProps));
	sync->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::debug);

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(1000000));
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(atlcamwithaffinewithmasknosync, *boost::unit_test::disabled())
{
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);

	NvV4L2CameraProps nvCamProps(400, 400, 2);
	nvCamProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(nvCamProps));

	NvTransformProps nvprops(ImageMetadata::YUV444);
	nvprops.qlen = 1;
	nvprops.logHealth = true;
	nvprops.logHealthFrequency = 100;
	nvprops.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto transform = boost::shared_ptr<Module>(new NvTransform(nvprops));
	source->setNext(transform);

	auto stream = cudastream_sp(new ApraCudaStream);
	AffineTransformProps affineProps(AffineTransformProps::CUBIC, stream, 0,1792, 0, 0, 1);
	affineProps.qlen = 1;
	affineProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto m_affineTransform = boost::shared_ptr<AffineTransform>(new AffineTransform(affineProps));
	transform->setNext(m_affineTransform);

	MaskNPPIProps maskProps(200, 200, 100, MaskNPPIProps::AVAILABLE_MASKS::NONE, stream);
	maskProps.qlen = 1;
	maskProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto m_mask = boost::shared_ptr<MaskNPPI>(new MaskNPPI(maskProps));
	m_affineTransform->setNext(m_mask);

	// auto sync = boost::shared_ptr<CudaStreamSynchronize>(new CudaStreamSynchronize(CudaStreamSynchronizeProps(stream)));
	// m_mask->setNext(sync);

	EglRendererProps eglProps(0, 0);
	eglProps.qlen = 1;
	eglProps.fps = 60;
	eglProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	eglProps.logHealth = true;
	eglProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new EglRenderer(eglProps));
	m_mask->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::debug);

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(1000000));
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(atlcam1, *boost::unit_test::disabled())
{
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);
	NvV4L2CameraProps nvCamProps(400, 400, 2);
	nvCamProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;

	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(nvCamProps));

	// auto copySource = boost::shared_ptr<Module>(new DMAFDToHostCopy);
	// source->setNext(copySource);

	// auto fileWriter1 = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/nvv4l2/Frame_????.raw")));
	// copySource->setNext(fileWriter1);

	NvTransformProps nvprops(ImageMetadata::RGBA);
	nvprops.qlen = 1;
	nvprops.logHealth = true;
	nvprops.logHealthFrequency = 100;
	nvprops.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto transform = boost::shared_ptr<Module>(new NvTransform(nvprops));
	source->setNext(transform);

	// auto copySource1 = boost::shared_ptr<Module>(new DMAFDToHostCopy);
	// transform->setNext(copySource1);

	// auto fileWriter11 = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/nvv4l2232/Frame2_????.raw")));
	// copySource1->setNext(fileWriter11);

	EglRendererProps eglProps(0, 0);
	eglProps.qlen = 1;
	eglProps.fps = 60;
	eglProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	eglProps.logHealth = true;
	eglProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new EglRenderer(eglProps));
	transform->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::debug);

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(1000000));
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(windowCreateDestroy, *boost::unit_test::disabled())
{
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);
	NvV4L2CameraProps nvCamProps(640, 480, 3);
	nvCamProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(nvCamProps));

	NvTransformProps nvprops(ImageMetadata::RGBA);
	nvprops.qlen = 1;
	nvprops.logHealth = true;
	nvprops.logHealthFrequency = 100;
	nvprops.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto transform = boost::shared_ptr<Module>(new NvTransform(nvprops));
	source->setNext(transform);

	EglRendererProps eglProps(0, 0);
	eglProps.qlen = 1;
	eglProps.fps = 60;
	eglProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	eglProps.logHealth = true;
	eglProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<EglRenderer>(new EglRenderer(eglProps));
	transform->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());
	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(10));
	while(true)
	{
		sink->closeWindow();
		usleep(500000);
		sink->createWindow(640,480);
	}
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);

	p.stop();
	p.term();

	p.wait_for_all();
}
BOOST_AUTO_TEST_SUITE_END()
