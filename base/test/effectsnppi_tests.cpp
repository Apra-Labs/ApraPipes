#include "stdafx.h"
#include <boost/test/unit_test.hpp>

#include "FileReaderModule.h"
#include "ExternalSourceModule.h"
#include "ExternalSinkModule.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "AIPExceptions.h"
#include "CudaMemCopy.h"
#include "EffectsNPPI.h"
#include "JPEGEncoderNVJPEG.h"
#include "JPEGDecoderNVJPEG.h"
#include "CudaStreamSynchronize.h"
#include "EffectsKernel.h"

#include <chrono>

using sys_clock = std::chrono::system_clock;


#include "test_utils.h"

BOOST_AUTO_TEST_SUITE(effectsnppi_tests)

BOOST_AUTO_TEST_CASE(mono_1920x1080)
{
	auto width = 1920;
	auto height = 1080;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/mono_1920x1080.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::MONO, CV_8UC1, 0, CV_8U, FrameMetadata::HOST, true));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy1 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	fileReader->setNext(copy1);

	EffectsNPPIProps effectsProps(stream);
	effectsProps.brightness = 0;
	effectsProps.contrast = 1;
	auto effects = boost::shared_ptr<EffectsNPPI>(new EffectsNPPI(effectsProps));
	copy1->setNext(effects);
	auto copy2 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyDeviceToHost, stream)));
	effects->setNext(copy2);
	auto outputPinId = copy2->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE)[0];


	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	copy2->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(copy1->init());
	BOOST_TEST(effects->init());
	BOOST_TEST(copy2->init());
	BOOST_TEST(sink->init());

	{
		// no effects
		fileReader->step();
		copy1->step();
		effects->step();
		copy2->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(outputPinId) != frames.end()));
		auto outFrame = frames[outputPinId];
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);

		auto filename = "./data/mono_1920x1080.raw";
		Test_Utils::saveOrCompare(filename, (const uint8_t *)outFrame->data(), outFrame->size(), 0);
	}


	effectsProps.brightness = 30;

	for (auto i = 0; i < 5; i++)
	{
		effectsProps.brightness += 20;
		effects->setProps(effectsProps);
		effects->step();

		fileReader->step();
		copy1->step();
		effects->step();
		copy2->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(outputPinId) != frames.end()));
		auto outFrame = frames[outputPinId];
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);

		auto filename = "./data/testOutput/effects/brightness_mono_1920x1080/effectsnppi_tests_mono_1920x1080_to_brightness_" + std::to_string(effectsProps.brightness) + ".raw";
		Test_Utils::saveOrCompare(filename.c_str(), (const uint8_t *)outFrame->data(), outFrame->size(), 0);		
	}

	effectsProps.brightness = 10;
	
	for (auto i = 0; i < 5; i++)
	{
		effectsProps.brightness -= 20;
		effects->setProps(effectsProps);
		effects->step();

		fileReader->step();
		copy1->step();
		effects->step();
		copy2->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(outputPinId) != frames.end()));
		auto outFrame = frames[outputPinId];
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);

		auto filename = "./data/testOutput/effects/brightness_mono_1920x1080/effectsnppi_tests_mono_1920x1080_to_brightness_sub_" + std::to_string(effectsProps.brightness*-1) + ".raw";
		Test_Utils::saveOrCompare(filename.c_str(), (const uint8_t *)outFrame->data(), outFrame->size(), 0);
	}

	effectsProps.brightness = 0;
	effectsProps.contrast = 1;

	for (auto i = 0; i < 5; i++)
	{
		effectsProps.contrast += 1;
		effects->setProps(effectsProps);
		effects->step();

		fileReader->step();
		copy1->step();
		effects->step();
		copy2->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(outputPinId) != frames.end()));
		auto outFrame = frames[outputPinId];
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);

		auto filename = "./data/testOutput/effects/contrast_mono_1920x1080/effectsnppi_tests_mono_1920x1080_to_contrast_" + std::to_string(effectsProps.contrast) + ".raw";
		Test_Utils::saveOrCompare(filename.c_str(), (const uint8_t *)outFrame->data(), outFrame->size(), 0);		
	}

	std::vector<double> contrastValues = { 1.0 / 2, 1.0 / 3, 1.0 / 4, 1.0 / 5, 1.0 / 6 };

	for (auto i = 0; i < 5; i++)
	{
		effectsProps.contrast = contrastValues[i];
		effects->setProps(effectsProps);
		effects->step();

		fileReader->step();
		copy1->step();
		effects->step();
		copy2->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(outputPinId) != frames.end()));
		auto outFrame = frames[outputPinId];
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);

		auto filename = "./data/testOutput/effects/contrast_mono_1920x1080/effectsnppi_tests_mono_1920x1080_to_contrast_div_" + std::to_string(effectsProps.contrast) + ".raw";
		Test_Utils::saveOrCompare(filename.c_str(), (const uint8_t *)outFrame->data(), outFrame->size(), 0);
	}


	effectsProps.brightness = 0;
	effectsProps.contrast = 1;

	for (auto i = 0; i < 5; i++)
	{
		effectsProps.brightness -= 10;
		effectsProps.contrast += 1;
		effects->setProps(effectsProps);
		effects->step();

		fileReader->step();
		copy1->step();
		effects->step();
		copy2->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(outputPinId) != frames.end()));
		auto outFrame = frames[outputPinId];
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);

		auto filename = "./data/testOutput/effects/brightness_contrast_mono_1920x1080/effectsnppi_tests_mono_1920x1080_to_brightness_sub_" + std::to_string(effectsProps.brightness*-1) + "_contrast_" + std::to_string(effectsProps.contrast) + ".raw";
		Test_Utils::saveOrCompare(filename.c_str(), (const uint8_t *)outFrame->data(), outFrame->size(), 0);
	}
	
	effectsProps.brightness = 0;

	for (auto i = 0; i < 5; i++)
	{
		effectsProps.brightness += 10;
		effectsProps.contrast = contrastValues[i];
		effects->setProps(effectsProps);
		effects->step();

		fileReader->step();
		copy1->step();
		effects->step();
		copy2->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(outputPinId) != frames.end()));
		auto outFrame = frames[outputPinId];
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);

		auto filename = "./data/testOutput/effects/brightness_contrast_mono_1920x1080/effectsnppi_tests_mono_1920x1080_to_brightness_" + std::to_string(effectsProps.brightness) + "_contrast_" + std::to_string(effectsProps.contrast) + ".raw";
		Test_Utils::saveOrCompare(filename.c_str(), (const uint8_t *)outFrame->data(), outFrame->size(), 0);
	}

}

BOOST_AUTO_TEST_CASE(yuv420_640x360)
{
	auto width = 640;
	auto height = 360;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/yuv420_640x360.raw")));
	auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy1 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	fileReader->setNext(copy1);

	EffectsNPPIProps effectsProps(stream);
	effectsProps.brightness = 0;
	effectsProps.contrast = 1;
	auto effects = boost::shared_ptr<EffectsNPPI>(new EffectsNPPI(effectsProps));
	copy1->setNext(effects);
	
	auto encoder = boost::shared_ptr<JPEGEncoderNVJPEG>(new JPEGEncoderNVJPEG(JPEGEncoderNVJPEGProps(stream)));
	effects->setNext(encoder);
	auto outputPinId = encoder->getAllOutputPinsByType(FrameMetadata::ENCODED_IMAGE)[0];

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	encoder->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(copy1->init());
	BOOST_TEST(effects->init());
	BOOST_TEST(encoder->init());
	BOOST_TEST(sink->init());

	{
		// no effects
		fileReader->step();
		copy1->step();
		effects->step();
		encoder->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(outputPinId) != frames.end()));
		auto outFrame = frames[outputPinId];
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

		auto filename = "./data/testOutput/effectsnppi_tests_yuv420_640x360_noeffects.jpg";
		Test_Utils::saveOrCompare(filename, (const uint8_t *)outFrame->data(), outFrame->size(), 0);
	}

	effectsProps.brightness = 30;

	for (auto i = 0; i < 5; i++)
	{
		effectsProps.brightness += 20;
		effects->setProps(effectsProps);
		effects->step();

		fileReader->step();
		copy1->step();
		effects->step();
		encoder->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(outputPinId) != frames.end()));
		auto outFrame = frames[outputPinId];
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

		auto filename = "./data/testOutput/effects/brightness_yuv420_640x360/effectsnppi_tests_yuv420_640x360_to_brightness_" + std::to_string(effectsProps.brightness) + ".jpg";
		Test_Utils::saveOrCompare(filename.c_str(), (const uint8_t *)outFrame->data(), outFrame->size(), 0);
	}

	effectsProps.brightness = 10;

	for (auto i = 0; i < 5; i++)
	{
		effectsProps.brightness -= 20;
		effects->setProps(effectsProps);
		effects->step();

		fileReader->step();
		copy1->step();
		effects->step();
		encoder->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(outputPinId) != frames.end()));
		auto outFrame = frames[outputPinId];
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

		auto filename = "./data/testOutput/effects/brightness_yuv420_640x360/effectsnppi_tests_yuv420_640x360_to_brightness_sub_" + std::to_string(effectsProps.brightness*-1) + ".jpg";
		Test_Utils::saveOrCompare(filename.c_str(), (const uint8_t *)outFrame->data(), outFrame->size(), 0);
	}

	effectsProps.brightness = 0;
	effectsProps.contrast = 1;

	for (auto i = 0; i < 5; i++)
	{
		effectsProps.contrast += 1;
		effects->setProps(effectsProps);
		effects->step();

		fileReader->step();
		copy1->step();
		effects->step();
		encoder->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(outputPinId) != frames.end()));
		auto outFrame = frames[outputPinId];
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

		auto filename = "./data/testOutput/effects/contrast_yuv420_640x360/effectsnppi_tests_yuv420_640x360_to_contrast_" + std::to_string(effectsProps.contrast) + ".jpg";
		Test_Utils::saveOrCompare(filename.c_str(), (const uint8_t *)outFrame->data(), outFrame->size(), 0);
	}

	std::vector<double> contrastValues = { 1.0 / 2, 1.0 / 3, 1.0 / 4, 1.0 / 5, 1.0 / 6 };

	for (auto i = 0; i < 5; i++)
	{
		effectsProps.contrast = contrastValues[i];
		effects->setProps(effectsProps);
		effects->step();

		fileReader->step();
		copy1->step();
		effects->step();
		encoder->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(outputPinId) != frames.end()));
		auto outFrame = frames[outputPinId];
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

		auto filename = "./data/testOutput/effects/contrast_yuv420_640x360/effectsnppi_tests_yuv420_640x360_to_contrast_div_" + std::to_string(effectsProps.contrast) + ".jpg";
		Test_Utils::saveOrCompare(filename.c_str(), (const uint8_t *)outFrame->data(), outFrame->size(), 0);
	}

	effectsProps.brightness = 0;
	effectsProps.contrast = 1;

	for (auto i = 0; i < 5; i++)
	{
		effectsProps.brightness -= 10;
		effectsProps.contrast += 1;
		effects->setProps(effectsProps);
		effects->step();

		fileReader->step();
		copy1->step();
		effects->step();
		encoder->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(outputPinId) != frames.end()));
		auto outFrame = frames[outputPinId];
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

		auto filename = "./data/testOutput/effects/brightness_contrast_yuv420_640x360/effectsnppi_tests_yuv420_640x360_to_brightness_sub_" + std::to_string(effectsProps.brightness*-1) + "_contrast_" + std::to_string(effectsProps.contrast) + ".jpg";
		Test_Utils::saveOrCompare(filename.c_str(), (const uint8_t *)outFrame->data(), outFrame->size(), 0);
	}

	effectsProps.brightness = 0;

	for (auto i = 0; i < 5; i++)
	{
		effectsProps.brightness += 10;
		effectsProps.contrast = contrastValues[i];
		effects->setProps(effectsProps);
		effects->step();

		fileReader->step();
		copy1->step();
		effects->step();
		encoder->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(outputPinId) != frames.end()));
		auto outFrame = frames[outputPinId];
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

		auto filename = "./data/testOutput/effects/brightness_contrast_yuv420_640x360/effectsnppi_tests_yuv420_640x360_to_brightness_" + std::to_string(effectsProps.brightness) + "_contrast_" + std::to_string(effectsProps.contrast) + ".jpg";
		Test_Utils::saveOrCompare(filename.c_str(), (const uint8_t *)outFrame->data(), outFrame->size(), 0);
	}


	effectsProps.brightness = 0;
	effectsProps.contrast = 1;

	effectsProps.hue = -110;

	for (auto i = 0; i < 11; i++)
	{
		effectsProps.hue += 10;
		effects->setProps(effectsProps);
		effects->step();

		fileReader->step();
		copy1->step();
		effects->step();
		encoder->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(outputPinId) != frames.end()));
		auto outFrame = frames[outputPinId];
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

		auto prefix = std::to_string(i);
		if (i < 10)
		{
			prefix = "0" + prefix;
		}
		auto filename = "./data/testOutput/effects/hue_yuv420_640x360/" + prefix + "effectsnppi_tests_yuv420_640x360_to_hue_sub_" + std::to_string(effectsProps.hue*-1) + ".jpg";
		Test_Utils::saveOrCompare(filename.c_str(), (const uint8_t *)outFrame->data(), outFrame->size(), 0);
	}

	effectsProps.hue = 0;

	for (auto i = 11; i < 20; i++)
	{
		effectsProps.hue += 10;		
		effects->setProps(effectsProps);
		effects->step();

		fileReader->step();
		copy1->step();
		effects->step();
		encoder->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(outputPinId) != frames.end()));
		auto outFrame = frames[outputPinId];
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

		auto filename = "./data/testOutput/effects/hue_yuv420_640x360/" + std::to_string(i) + "effectsnppi_tests_yuv420_640x360_to_hue_" + std::to_string(effectsProps.hue) + ".jpg";
		Test_Utils::saveOrCompare(filename.c_str(), (const uint8_t *)outFrame->data(), outFrame->size(), 0);
	}

	effectsProps.hue = 0;

	for (auto i = 0; i < 20; i++)
	{
		effectsProps.saturation += 0.5;
		effects->setProps(effectsProps);
		effects->step();

		fileReader->step();
		copy1->step();
		effects->step();
		encoder->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(outputPinId) != frames.end()));
		auto outFrame = frames[outputPinId];
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

		auto filename = "./data/testOutput/effects/saturation_yuv420_640x360/" + std::to_string(effectsProps.saturation) + "effectsnppi_tests_yuv420_640x360_to_saturation.jpg";
		Test_Utils::saveOrCompare(filename.c_str(), (const uint8_t *)outFrame->data(), outFrame->size(), 0);
	}

	effectsProps.saturation = 1.1;

	for (auto i = 20; i < 31; i++)
	{
		effectsProps.saturation -= 0.1;
		effects->setProps(effectsProps);
		effects->step();

		fileReader->step();
		copy1->step();
		effects->step();
		encoder->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(outputPinId) != frames.end()));
		auto outFrame = frames[outputPinId];
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

		auto filename = "./data/testOutput/effects/saturation_yuv420_640x360/" + std::to_string(effectsProps.saturation) + "effectsnppi_tests_yuv420_640x360_to_saturation.jpg";
		Test_Utils::saveOrCompare(filename.c_str(), (const uint8_t *)outFrame->data(), outFrame->size(), 0);
	}

	effectsProps.saturation = 1;

	for (auto i = 0; i < 5; i++)
	{
		effectsProps.hue -= 10;
		effectsProps.saturation += 1;
		effects->setProps(effectsProps);
		effects->step();

		fileReader->step();
		copy1->step();
		effects->step();
		encoder->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(outputPinId) != frames.end()));
		auto outFrame = frames[outputPinId];
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

		auto filename = "./data/testOutput/effects/hue_saturation_yuv420_640x360/" + std::to_string(i) + "effectsnppi_tests_yuv420_640x360_to_hue_sub_" + std::to_string(effectsProps.hue*-1) + "_saturation_" + std::to_string(effectsProps.saturation) + ".jpg";
		Test_Utils::saveOrCompare(filename.c_str(), (const uint8_t *)outFrame->data(), outFrame->size(), 0);
	}

	effectsProps.hue = 0;
	effectsProps.saturation = 1;

	for (auto i = 5; i < 10; i++)
	{
		effectsProps.hue += 10;
		effectsProps.contrast -= 0.1;
		effects->setProps(effectsProps);
		effects->step();

		fileReader->step();
		copy1->step();
		effects->step();
		encoder->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(outputPinId) != frames.end()));
		auto outFrame = frames[outputPinId];
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

		auto filename = "./data/testOutput/effects/hue_saturation_yuv420_640x360/" + std::to_string(i) + "effectsnppi_tests_yuv420_640x360_to_hue_" + std::to_string(effectsProps.hue) + "_saturation_" + std::to_string(effectsProps.saturation) + ".jpg";
		Test_Utils::saveOrCompare(filename.c_str(), (const uint8_t *)outFrame->data(), outFrame->size(), 0);
	}
}

BOOST_AUTO_TEST_CASE(yuv420_1920x1080_performance, *boost::unit_test::disabled())
{
	auto width = 1920;
	auto height = 1080;

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/yuv420_1920x1080.raw")));
	auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copy1 = boost::shared_ptr<Module>(new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream)));
	fileReader->setNext(copy1);

	auto externalSink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	copy1->setNext(externalSink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(copy1->init());
	BOOST_TEST(externalSink->init());

	fileReader->step();
	copy1->step();
	auto frames = externalSink->try_pop();
	BOOST_TEST(frames.size() == 1);

	auto frame = frames.begin()->second;

	auto externalSource = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto cudaImageMetadata = frame->getMetadata();
	auto pin = externalSource->addOutputPin(cudaImageMetadata);

	EffectsNPPIProps effectsProps(stream);
	effectsProps.brightness = 20;
	effectsProps.contrast = 2;
	effectsProps.hue = 20;
	effectsProps.saturation = 2;
	effectsProps.logHealth = true;
	auto effects = boost::shared_ptr<EffectsNPPI>(new EffectsNPPI(effectsProps));
	externalSource->setNext(effects);

	auto sync = boost::shared_ptr<Module>(new CudaStreamSynchronize(CudaStreamSynchronizeProps(stream)));
	effects->setNext(sync);

	BOOST_TEST(externalSource->init());
	BOOST_TEST(effects->init());
	BOOST_TEST(sync->init());

	frame_container cudaFrames;
	cudaFrames.insert(make_pair(pin, frame));

	for (auto i = 0; i < 10000; i++)
	{
		externalSource->send(cudaFrames);
		effects->step();
		sync->step();
	}

	BOOST_TEST(fileReader->term());
	BOOST_TEST(copy1->term());
	BOOST_TEST(externalSource->term());
	BOOST_TEST(effects->term());
	BOOST_TEST(sync->term());
}

BOOST_AUTO_TEST_CASE(kernel_test, *boost::unit_test::disabled())
{
	int width = 1920;
	int height = 1080;
	int width_2 = width >> 1;
	int height_2 = height >> 1;
	int step = 2048;
	int step_2 = 1024;

	int size = step * height;
	int size_2 = step_2 * height_2;

	void *y, *u, *v, *Y, *U, *V;
	cudaMalloc(&y, size);
	cudaMalloc(&u, size_2);
	cudaMalloc(&v, size_2);
	cudaMalloc(&Y, size);
	cudaMalloc(&U, size_2);
	cudaMalloc(&V, size_2);

	cudaMemset(y, 30, size);
	cudaMemset(u, 70, size_2);
	cudaMemset(v, 200, size_2);
	cudaMemset(Y, 0, size);
	cudaMemset(U, 128, size_2);
	cudaMemset(V, 128, size_2);

	cudaDeviceSynchronize();

	NppiSize dims = { width, height };

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	double totalTime = 0;
	for (auto i = 0; i < 10; i++)
	{
		auto start = sys_clock::now();
		for (auto j = 0; j < 1000; j++)
		{
			launchYUV420Effects(const_cast<const Npp8u *>(static_cast<Npp8u*>(y)),
				const_cast<const Npp8u *>(static_cast<Npp8u*>(u)),
				const_cast<const Npp8u *>(static_cast<Npp8u*>(v)),
				static_cast<Npp8u*>(Y),
				static_cast<Npp8u*>(U),
				static_cast<Npp8u*>(V),
				20, 2, 20, 2,
				step, step_2, dims, stream);
			cudaStreamSynchronize(stream);
		}
		auto end = sys_clock::now();
		auto diff = end - start;
		totalTime += diff.count();

		auto timeElapsed = diff.count() / 10000000.0;
		double fps = 1000.0 / timeElapsed;

		std::cout << "Processed 1000. Time Elapsed<" << timeElapsed << "> fps<" << static_cast<int>(fps) << ">" << std::endl;
	}

	totalTime = totalTime / 1000000000.0;
	auto avgTime = totalTime / (10);
	double fps = 10 * 1000.0 / (totalTime);

	std::cout << "AvgTime<" << avgTime << "> fps<" << static_cast<int>(fps) << ">" << std::endl;
	
	cudaStreamDestroy(stream);

	cudaFree(y);
	cudaFree(u);
	cudaFree(v);
	cudaFree(Y);
	cudaFree(U);
	cudaFree(V);
}

BOOST_AUTO_TEST_CASE(img_864x576)
{
	auto width = 864;
	auto height = 576;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/img_864x576.jpg")));
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));

	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto decoder = boost::shared_ptr<JPEGDecoderNVJPEG>(new JPEGDecoderNVJPEG(JPEGDecoderNVJPEGProps(stream)));
	fileReader->setNext(decoder);

	EffectsNPPIProps effectsProps(stream);
	effectsProps.brightness = 0;
	effectsProps.contrast = 1;
	auto effects = boost::shared_ptr<EffectsNPPI>(new EffectsNPPI(effectsProps));
	decoder->setNext(effects);

	auto encoder = boost::shared_ptr<JPEGEncoderNVJPEG>(new JPEGEncoderNVJPEG(JPEGEncoderNVJPEGProps(stream)));
	effects->setNext(encoder);
	auto outputPinId = encoder->getAllOutputPinsByType(FrameMetadata::ENCODED_IMAGE)[0];

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	encoder->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(decoder->init());
	BOOST_TEST(effects->init());
	BOOST_TEST(encoder->init());
	BOOST_TEST(sink->init());

	{
		// no effects
		fileReader->step();
		decoder->step();
		effects->step();
		encoder->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(outputPinId) != frames.end()));
		auto outFrame = frames[outputPinId];
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

		auto filename = "./data/testOutput/effectsnppi_tests_img_864x576_noeffects.jpg";
		Test_Utils::saveOrCompare(filename, (const uint8_t *)outFrame->data(), outFrame->size(), 0);
	}

	effectsProps.brightness = 30;

	for (auto i = 0; i < 5; i++)
	{
		effectsProps.brightness += 20;
		effects->setProps(effectsProps);
		effects->step();

		fileReader->step();
		decoder->step();
		effects->step();
		encoder->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(outputPinId) != frames.end()));
		auto outFrame = frames[outputPinId];
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

		auto filename = "./data/testOutput/effects/brightness_img_864x576/effectsnppi_tests_img_864x576_to_brightness_" + std::to_string(effectsProps.brightness) + ".jpg";
		Test_Utils::saveOrCompare(filename.c_str(), (const uint8_t *)outFrame->data(), outFrame->size(), 0);
	}

	effectsProps.brightness = 10;

	for (auto i = 0; i < 5; i++)
	{
		effectsProps.brightness -= 20;
		effects->setProps(effectsProps);
		effects->step();

		fileReader->step();
		decoder->step();
		effects->step();
		encoder->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(outputPinId) != frames.end()));
		auto outFrame = frames[outputPinId];
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

		auto filename = "./data/testOutput/effects/brightness_img_864x576/effectsnppi_tests_img_864x576_to_brightness_sub_" + std::to_string(effectsProps.brightness*-1) + ".jpg";
		Test_Utils::saveOrCompare(filename.c_str(), (const uint8_t *)outFrame->data(), outFrame->size(), 0);
	}

	effectsProps.brightness = 0;
	effectsProps.contrast = 1;

	for (auto i = 0; i < 5; i++)
	{
		effectsProps.contrast += 1;
		effects->setProps(effectsProps);
		effects->step();

		fileReader->step();
		decoder->step();
		effects->step();
		encoder->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(outputPinId) != frames.end()));
		auto outFrame = frames[outputPinId];
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

		auto filename = "./data/testOutput/effects/contrast_img_864x576/effectsnppi_tests_img_864x576_to_contrast_" + std::to_string(effectsProps.contrast) + ".jpg";
		Test_Utils::saveOrCompare(filename.c_str(), (const uint8_t *)outFrame->data(), outFrame->size(), 0);
	}

	std::vector<double> contrastValues = { 1.0 / 2, 1.0 / 3, 1.0 / 4, 1.0 / 5, 1.0 / 6 };

	for (auto i = 0; i < 5; i++)
	{
		effectsProps.contrast = contrastValues[i];
		effects->setProps(effectsProps);
		effects->step();

		fileReader->step();
		decoder->step();
		effects->step();
		encoder->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(outputPinId) != frames.end()));
		auto outFrame = frames[outputPinId];
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

		auto filename = "./data/testOutput/effects/contrast_img_864x576/effectsnppi_tests_img_864x576_to_contrast_div_" + std::to_string(effectsProps.contrast) + ".jpg";
		Test_Utils::saveOrCompare(filename.c_str(), (const uint8_t *)outFrame->data(), outFrame->size(), 0);
	}

	effectsProps.brightness = 0;
	effectsProps.contrast = 1;

	for (auto i = 0; i < 5; i++)
	{
		effectsProps.brightness -= 10;
		effectsProps.contrast += 1;
		effects->setProps(effectsProps);
		effects->step();

		fileReader->step();
		decoder->step();
		effects->step();
		encoder->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(outputPinId) != frames.end()));
		auto outFrame = frames[outputPinId];
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

		auto filename = "./data/testOutput/effects/brightness_contrast_img_864x576/effectsnppi_tests_img_864x576_to_brightness_sub_" + std::to_string(effectsProps.brightness*-1) + "_contrast_" + std::to_string(effectsProps.contrast) + ".jpg";
		Test_Utils::saveOrCompare(filename.c_str(), (const uint8_t *)outFrame->data(), outFrame->size(), 0);
	}

	effectsProps.brightness = 0;

	for (auto i = 0; i < 5; i++)
	{
		effectsProps.brightness += 10;
		effectsProps.contrast = contrastValues[i];
		effects->setProps(effectsProps);
		effects->step();

		fileReader->step();
		decoder->step();
		effects->step();
		encoder->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(outputPinId) != frames.end()));
		auto outFrame = frames[outputPinId];
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

		auto filename = "./data/testOutput/effects/brightness_contrast_img_864x576/effectsnppi_tests_img_864x576_to_brightness_" + std::to_string(effectsProps.brightness) + "_contrast_" + std::to_string(effectsProps.contrast) + ".jpg";
		Test_Utils::saveOrCompare(filename.c_str(), (const uint8_t *)outFrame->data(), outFrame->size(), 0);
	}


	effectsProps.brightness = 0;
	effectsProps.contrast = 1;

	effectsProps.hue = -10;

	for (auto i = 0; i < 27; i++)
	{
		effectsProps.hue += 10;
		if (effectsProps.hue > 255)
		{
			effectsProps.hue = 255;
		}
		effects->setProps(effectsProps);
		effects->step();

		fileReader->step();
		decoder->step();
		effects->step();
		encoder->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(outputPinId) != frames.end()));
		auto outFrame = frames[outputPinId];
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

		auto prefix = std::to_string(i);
		if (i < 10)
		{
			prefix = "0" + prefix;
		}
		auto filename = "./data/testOutput/effects/hue_img_864x576/" + prefix + "effectsnppi_tests_img_864x576_to_hue_" + std::to_string(effectsProps.hue) + ".jpg";
		Test_Utils::saveOrCompare(filename.c_str(), (const uint8_t *)outFrame->data(), outFrame->size(), 0);
	}
	

	effectsProps.hue = 0;
	effectsProps.saturation = -0.1;

	for (auto i = 0; i < 40; i++)
	{
		effectsProps.saturation += 0.1;
		effects->setProps(effectsProps);
		effects->step();

		fileReader->step();
		decoder->step();
		effects->step();
		encoder->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(outputPinId) != frames.end()));
		auto outFrame = frames[outputPinId];
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

		auto filename = "./data/testOutput/effects/saturation_img_864x576/" + std::to_string(effectsProps.saturation) + "effectsnppi_tests_img_864x576_to_saturation.jpg";
		Test_Utils::saveOrCompare(filename.c_str(), (const uint8_t *)outFrame->data(), outFrame->size(), 0);
	}

	effectsProps.saturation = 1;

	for (auto i = 0; i < 5; i++)
	{
		effectsProps.hue -= 10;
		effectsProps.saturation += 1;
		effects->setProps(effectsProps);
		effects->step();

		fileReader->step();
		decoder->step();
		effects->step();
		encoder->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(outputPinId) != frames.end()));
		auto outFrame = frames[outputPinId];
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

		auto filename = "./data/testOutput/effects/hue_saturation_img_864x576/" + std::to_string(i) + "effectsnppi_tests_img_864x576_to_hue_sub_" + std::to_string(effectsProps.hue*-1) + "_saturation_" + std::to_string(effectsProps.saturation) + ".jpg";
		Test_Utils::saveOrCompare(filename.c_str(), (const uint8_t *)outFrame->data(), outFrame->size(), 0);
	}

	effectsProps.hue = 0;
	effectsProps.saturation = 1;

	for (auto i = 5; i < 10; i++)
	{
		effectsProps.hue += 10;
		effectsProps.contrast -= 0.1;
		effects->setProps(effectsProps);
		effects->step();

		fileReader->step();
		decoder->step();
		effects->step();
		encoder->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(outputPinId) != frames.end()));
		auto outFrame = frames[outputPinId];
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::ENCODED_IMAGE);

		auto filename = "./data/testOutput/effects/hue_saturation_img_864x576/" + std::to_string(i) + "effectsnppi_tests_img_864x576_to_hue_" + std::to_string(effectsProps.hue) + "_saturation_" + std::to_string(effectsProps.saturation) + ".jpg";
		Test_Utils::saveOrCompare(filename.c_str(), (const uint8_t *)outFrame->data(), outFrame->size(), 0);
	}
}

BOOST_AUTO_TEST_SUITE_END()
