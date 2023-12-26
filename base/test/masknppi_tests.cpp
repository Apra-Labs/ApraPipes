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

using sys_clock = std::chrono::system_clock;

#include "test_utils.h"

BOOST_AUTO_TEST_SUITE(masknppi_tests)

BOOST_AUTO_TEST_CASE(yuyv)
{
	auto width = 640;
	auto height = 480;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/testOutput/nvv4l2/frame_0118.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::UYVY, CV_8UC2, 0, CV_8U, FrameMetadata::HOST, true));
	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy1 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	fileReader->setNext(copy1);

	MaskNPPIProps maskProps(640, 480, 100, stream);
	auto circMask = boost::shared_ptr<MaskNPPI>(new MaskNPPI(maskProps));
	copy1->setNext(circMask);

	auto copy2 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyDeviceToHost, stream)));
	circMask->setNext(copy2);
	auto outputPinId = copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];

	// auto sink = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/MAskFrame???.raw")));
	// copy2->setNext(sink);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	copy2->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(copy1->init());
	BOOST_TEST(circMask->init());
	BOOST_TEST(copy2->init());
	BOOST_TEST(sink->init());

	fileReader->step();
	copy1->step();
	circMask->step();
	copy2->step();
	auto frames = sink->pop();
	BOOST_TEST((frames.find(outputPinId) != frames.end()));
	auto outFrame = frames[outputPinId];
	BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);

	auto filename = "./data/testOutput/mask_yuyv.raw";
	Test_Utils::saveOrCompare(filename, (const uint8_t *)outFrame->data(), outFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(yuyv_test)
{
	auto width = 640;
	auto height = 480;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/testOutput/nvv4l2/frame_0118.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::YUYV, CV_8UC2, 0, CV_8U, FrameMetadata::HOST, true));
	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy1 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	fileReader->setNext(copy1);

	MaskNPPIProps maskProps(640, 480, 100, stream);
	auto circMask = boost::shared_ptr<MaskNPPI>(new MaskNPPI(maskProps));
	copy1->setNext(circMask);

	auto copy2 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyDeviceToHost, stream)));
	circMask->setNext(copy2);

	auto sink = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/MAskFrame???.raw")));
	copy2->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(copy1->init());
	BOOST_TEST(circMask->init());
	BOOST_TEST(copy2->init());
	BOOST_TEST(sink->init());

	fileReader->step();
	copy1->step();
	circMask->step();
	copy2->step();
	sink->step();
}

BOOST_AUTO_TEST_CASE(rgba)
{
	auto width = 1920;
	auto height = 960;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/overlay_1920x960_BGRA.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::BGRA, CV_8UC4, 0, CV_8U, FrameMetadata::HOST, true));
	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy1 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	fileReader->setNext(copy1);

	MaskNPPIProps maskProps(640, 480, 100, stream);
	auto circMask = boost::shared_ptr<MaskNPPI>(new MaskNPPI(maskProps));
	copy1->setNext(circMask);

	auto copy2 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyDeviceToHost, stream)));
	circMask->setNext(copy2);

	auto sink = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./RGBA/MAskFrame???.raw")));
	copy2->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(copy1->init());
	BOOST_TEST(circMask->init());
	BOOST_TEST(copy2->init());
	BOOST_TEST(sink->init());

	fileReader->step();
	copy1->step();
	circMask->step();
	copy2->step();
	sink->step();
}

BOOST_AUTO_TEST_CASE(yuv444)
{
	auto width = 1280;
	auto height = 960;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/1280x960_444p.raw")));
	auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::YUV444, size_t(0), CV_8U, FrameMetadata::HOST));
	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy1 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	fileReader->setNext(copy1);

	MaskNPPIProps maskProps(640, 480, 100, stream);
	auto circMask = boost::shared_ptr<MaskNPPI>(new MaskNPPI(maskProps));
	copy1->setNext(circMask);

	auto copy2 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyDeviceToHost, stream)));
	circMask->setNext(copy2);

	auto sink = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./YUV444/MAskFrame???.raw")));
	copy2->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(copy1->init());
	BOOST_TEST(circMask->init());
	BOOST_TEST(copy2->init());
	BOOST_TEST(sink->init());

	fileReader->step();
	copy1->step();
	circMask->step();
	copy2->step();
	sink->step();
}

BOOST_AUTO_TEST_CASE(cam_mask)
{
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(NvV4L2CameraProps(400, 400, 2)));

	auto nv_transform = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::YUV444)));
	source->setNext(nv_transform);

	auto stream = cudastream_sp(new ApraCudaStream);
	
	MaskNPPIProps maskProps(320, 240, 100, stream);
	auto circMask = boost::shared_ptr<MaskNPPI>(new MaskNPPI(maskProps));
	nv_transform->setNext(circMask);


	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0)));
	circMask->setNext(sink);

	// auto dmaToHostCopy = boost::shared_ptr<Module>(new DMAFDToHostCopy);
	// sync->setNext(dmaToHostCopy);

	// auto fileWriter = boost::shared_ptr<FileWriterModule>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/MaskOutputImages/Frame???.raw")));
	// dmaToHostCopy->setNext(fileWriter);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(200));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all();


}

BOOST_AUTO_TEST_CASE(cam_mask2)
{
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(NvV4L2CameraProps(400, 400, 2)));

	auto nv_transform = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::YUV444)));
	source->setNext(nv_transform);

	auto stream = cudastream_sp(new ApraCudaStream);
	
	MaskNPPIProps maskProps(320, 240, 100, stream);
	auto circMask = boost::shared_ptr<MaskNPPI>(new MaskNPPI(maskProps));
	nv_transform->setNext(circMask);

	auto sync = boost::shared_ptr<Module>(new CudaStreamSynchronize(CudaStreamSynchronizeProps(stream)));
	circMask->setNext(sync);

	// auto nv_transform2 = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::RGBA)));
	// sync->setNext(nv_transform2);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0)));
	sync->setNext(sink);

	// auto dmaToHostCopy = boost::shared_ptr<Module>(new DMAFDToHostCopy);
	// sync->setNext(dmaToHostCopy);

	// auto fileWriter = boost::shared_ptr<FileWriterModule>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/MaskOutputImages/Frame???.raw")));
	// dmaToHostCopy->setNext(fileWriter);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(200));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);
	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(combineMaskTest)
{
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(NvV4L2CameraProps(400, 400, 2)));

	auto nv_transform = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::YUV444)));
	source->setNext(nv_transform);

	auto stream = cudastream_sp(new ApraCudaStream);
	
	MaskNPPIProps maskProps(200, 200, 100, stream);

	auto m_Mask = boost::shared_ptr<MaskNPPI>(new MaskNPPI(maskProps));
	nv_transform->setNext(m_Mask);

	auto sync = boost::shared_ptr<Module>(new CudaStreamSynchronize(CudaStreamSynchronizeProps(stream)));
	m_Mask->setNext(sync);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0)));
	sync->setNext(sink);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(10));
	auto currProps = m_Mask->getProps();
	currProps.maskSelected = MaskNPPIProps::AVAILABLE_MASKS::CIRCLE;
	m_Mask->setProps(currProps);

	boost::this_thread::sleep_for(boost::chrono::seconds(15));
	currProps = m_Mask->getProps();
	currProps.maskSelected = MaskNPPIProps::AVAILABLE_MASKS::OCTAGONAL;
	m_Mask->setProps(currProps);

	boost::this_thread::sleep_for(boost::chrono::seconds(20));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);
	p.stop();
	p.term();
	p.wait_for_all();
}
BOOST_AUTO_TEST_SUITE_END()
