#include <boost/test/unit_test.hpp>
#include "FileReaderModule.h"
#include "ExternalSinkModule.h"
#include "test_utils.h"
#include "PipeLine.h"
#include "ColorConversionXForm.h"

#ifdef ARM64
BOOST_AUTO_TEST_SUITE(color_conversion_tests, *boost::unit_test::disabled())
#else
BOOST_AUTO_TEST_SUITE(color_conversion_tests)
#endif

frame_sp colorConversion(std::string inputPathName, framemetadata_sp metadata, ColorConversionProps::ConversionType conversionType)
{
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps(inputPathName)));
	fileReader->addOutputPin(metadata);

	auto colorchange = boost::shared_ptr<ColorConversion>(new ColorConversion(ColorConversionProps(conversionType)));
	fileReader->setNext(colorchange);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	colorchange->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(colorchange->init());
	BOOST_TEST(sink->init());

	fileReader->step();
	colorchange->step();
	auto frames = sink->pop();
	BOOST_TEST(frames.size() == 1);
	auto outputFrame = frames.cbegin()->second;

	return outputFrame;
}

BOOST_AUTO_TEST_CASE(rgb_2_mono)
{
	std::string inputPathName = "./data/frame_1280x720_rgb.raw";
	auto metadata = framemetadata_sp(new RawImageMetadata(1280, 720, ImageMetadata::ImageType::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));
	auto conversionType = ColorConversionProps::ConversionType::RGB_TO_MONO;

	auto outputFrame = colorConversion(inputPathName, metadata, conversionType);

	BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/frame_1280x720_rgb_cc_mono.raw", const_cast<const uint8_t*>(static_cast<uint8_t*>(outputFrame->data())), outputFrame->size(), 0);
}

BOOST_AUTO_TEST_CASE(bgr_2_mono)
{
	std::string inputPathName = "./data/BGR_1080x720.raw";
	auto conversionType = ColorConversionProps::ConversionType::BGR_TO_MONO;
	auto metadata = framemetadata_sp(new RawImageMetadata(1080, 720, ImageMetadata::ImageType::BGR, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));

	auto outputFrame = colorConversion(inputPathName, metadata, conversionType);

	BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/frame_1080x720_bgr_cc_mono.raw", const_cast<const uint8_t*>(static_cast<uint8_t*>(outputFrame->data())), outputFrame->size(), 0);

}

BOOST_AUTO_TEST_CASE(bgr_2_rgb)
{
	std::string inputPathName = "./data/BGR_1080x720.raw";
	auto conversionType = ColorConversionProps::ConversionType::BGR_TO_RGB;
	auto metadata = framemetadata_sp(new RawImageMetadata(1080, 720, ImageMetadata::ImageType::BGR, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));

	auto outputFrame = colorConversion(inputPathName, metadata, conversionType);

	BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/frame_1080x720_bgr_cc_rgb.raw", const_cast<const uint8_t*>(static_cast<uint8_t*>(outputFrame->data())), outputFrame->size(), 0);

}

BOOST_AUTO_TEST_CASE(rgb_2_bgr)
{

	std::string inputPathName = "./data/frame_1280x720_rgb.raw";
	auto conversionType = ColorConversionProps::ConversionType::RGB_TO_BGR;
	auto metadata = framemetadata_sp(new RawImageMetadata(1280, 720, ImageMetadata::ImageType::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));

	auto outputFrame = colorConversion(inputPathName, metadata, conversionType);

	BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/frame_1280x720_rgb_cc_bgr.raw", const_cast<const uint8_t*>(static_cast<uint8_t*>(outputFrame->data())), outputFrame->size(), 0);

}

BOOST_AUTO_TEST_CASE(rgb_2_yuv420Planar)
{
	std::string inputPathName = "./data/frame_1280x720_rgb.raw";
	auto conversionType = ColorConversionProps::ConversionType::RGB_TO_YUV420PLANAR;
	auto metadata = framemetadata_sp(new RawImageMetadata(1280, 720, ImageMetadata::ImageType::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));

	auto outputFrame = colorConversion(inputPathName, metadata, conversionType);

	BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE_PLANAR);
	Test_Utils::saveOrCompare("./data/testOutput/frame_1280X720_RGB_cc_YUV420Planar.raw", const_cast<const uint8_t*>(static_cast<uint8_t*>(outputFrame->data())), outputFrame->size(), 0);

}

BOOST_AUTO_TEST_CASE(yuv420Planar_2_rgb)
{
	std::string inputPathName = "./data/YUV_420_planar.raw";
	auto conversionType = ColorConversionProps::ConversionType::YUV420PLANAR_TO_RGB;
	auto metadata = framemetadata_sp(new RawImagePlanarMetadata(1280, 720, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));

	auto outputFrame = colorConversion(inputPathName, metadata, conversionType);

	BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/frame_1280x720_YUV420Planar_cc_RGB.raw", const_cast<const uint8_t*>(static_cast<uint8_t*>(outputFrame->data())), outputFrame->size(), 0);

}

BOOST_AUTO_TEST_CASE(BayerBG8Bit_2_RGB)
{
	std::string inputPathName = "./data/Bayer_images/Rubiks_BayerBG8_800x800.raw";
	auto conversionType = ColorConversionProps::ConversionType::BAYERBG8_TO_RGB;
	auto metadata = framemetadata_sp(new RawImageMetadata(800, 800, ImageMetadata::ImageType::BAYERBG8, CV_8UC1, 0, CV_8U, FrameMetadata::HOST, true));

	auto outputFrame = colorConversion(inputPathName, metadata, conversionType);

	BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/frame_800x800_bayerBG8bit_cc_rgb.raw", const_cast<const uint8_t*>(static_cast<uint8_t*>(outputFrame->data())), outputFrame->size(), 0);

}

BOOST_AUTO_TEST_CASE(BayerBG8Bit_2_Mono)
{
	std::string inputPathName = "./data/Bayer_images/Rubiks_BayerBG8_800x800.raw";
	auto conversionType = ColorConversionProps::ConversionType::BAYERBG8_TO_MONO;
	auto metadata = framemetadata_sp(new RawImageMetadata(800, 800, ImageMetadata::ImageType::BAYERBG8, CV_8UC1, 0, CV_8U, FrameMetadata::HOST, true));

	auto outputFrame = colorConversion(inputPathName, metadata, conversionType);

	BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/frame_800x800_bayerBG8bit_cc_mono.raw", const_cast<const uint8_t*>(static_cast<uint8_t*>(outputFrame->data())), outputFrame->size(), 0);

}

BOOST_AUTO_TEST_CASE(BayerGB8Bit_2_RGB)
{
	std::string inputPathName = "./data/Bayer_images/Rubiks_bayerGB8_799xx800.raw";
	auto conversionType = ColorConversionProps::ConversionType::BAYERGB8_TO_RGB;
	auto metadata = framemetadata_sp(new RawImageMetadata(799, 800, ImageMetadata::ImageType::BAYERGB8, CV_8UC1, 0, CV_8U, FrameMetadata::HOST, true));

	auto outputFrame = colorConversion(inputPathName, metadata, conversionType);

	BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/frame_799x800_bayerGB8bit_cc_RGB.raw", const_cast<const uint8_t*>(static_cast<uint8_t*>(outputFrame->data())), outputFrame->size(), 0);

}

BOOST_AUTO_TEST_CASE(BayerGR8Bit_2_RGB)
{
	std::string inputPathName = "./data/Bayer_images/Rubiks_bayerGR8_800x799.raw";
	auto conversionType = ColorConversionProps::ConversionType::BAYERGR8_TO_RGB;
	auto metadata = framemetadata_sp(new RawImageMetadata(800, 799, ImageMetadata::ImageType::BAYERGR8, CV_8UC1, 0, CV_8U, FrameMetadata::HOST, true));

	auto outputFrame = colorConversion(inputPathName, metadata, conversionType);

	BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/frame_800x799_bayerGR8bit_cc_mono.raw", const_cast<const uint8_t*>(static_cast<uint8_t*>(outputFrame->data())), outputFrame->size(), 0);

}

BOOST_AUTO_TEST_CASE(BayerRG8Bit_2_RGB)
{
	std::string inputPathName = "./data/Bayer_images/Rubiks_bayerRG8_799x799.raw";
	auto conversionType = ColorConversionProps::ConversionType::BAYERRG8_TO_RGB;
	auto metadata = framemetadata_sp(new RawImageMetadata(799, 799, ImageMetadata::ImageType::BAYERRG8, CV_8UC1, 0, CV_8U, FrameMetadata::HOST, true));

	auto outputFrame = colorConversion(inputPathName, metadata, conversionType);

	BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
	Test_Utils::saveOrCompare("./data/testOutput/frame_799x799_bayerRG8bit_cc_RGB.raw", const_cast<const uint8_t*>(static_cast<uint8_t*>(outputFrame->data())), outputFrame->size(), 0);

}

BOOST_AUTO_TEST_SUITE_END()