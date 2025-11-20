#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include <filesystem>
#include <fstream>
#include <memory>
#include "ExternalSourceModule.h"
#include "ExternalSinkModule.h"
#include "ArchiveSpaceManager.h"
#include "Logger.h"
#include "AIPExceptions.h"
#include "test_utils.h"
#include "Module.h"
#include "PipeLine.h"

BOOST_AUTO_TEST_SUITE(archivespacemanager_tests)

void createDiskFiles()
{
	std::filesystem::create_directories(std::filesystem::path("data/archive/Cam1/2022/10/01"));
	std::filesystem::create_directories(std::filesystem::path("data/archive/Cam2/2022/10/01"));
	std::filesystem::create_directories(std::filesystem::path("data/archive/Cam3/2022/10/01"));
	std::ofstream a(std::filesystem::path("data/archive/Cam1/2022/10/01/file0.txt"));
	for (int i = 0; i < 10023; i++)
	{
		a << "a";
	}
	a.close();
	std::ofstream b(std::filesystem::path("data/archive/Cam2/2022/10/01/file0.txt"));
	for (int i = 0; i < 10023; i++)
	{
		b << "b";
	}
	b.close();
	std::ofstream c(std::filesystem::path("data/archive/Cam3/2022/10/01/file0.txt"));
	for (int i = 0; i < 10023; i++)
	{
		c << "c";
	}
	c.close();
};

BOOST_AUTO_TEST_CASE(basic)
{
	createDiskFiles();
	std::vector<std::string> folder = { "./data/archive" };
	Test_Utils::FileCleaner f(folder);
	uint32_t lowerWaterMark = 25000;
	uint32_t upperWaterMark = 30000;
	auto diskMan = std::shared_ptr<ArchiveSpaceManager>(new ArchiveSpaceManager(ArchiveSpaceManagerProps(lowerWaterMark, upperWaterMark, "./data/archive", 1)));
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
	auto pinId = diskMan->addOutputPin(metadata);
	auto sink = std::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	diskMan->setNext(sink);
	BOOST_TEST(diskMan->init());
	BOOST_TEST(sink->init());
	frame_container frames;
	diskMan->step();
	BOOST_ASSERT(diskMan->finalArchiveSpace < lowerWaterMark);

}

BOOST_AUTO_TEST_CASE(create_files)
{
	createDiskFiles();
	std::vector<std::string> folder = { "./data/archive" };
	Test_Utils::FileCleaner f(folder);
	uint32_t lowerWaterMark = 25000;
	uint32_t upperWaterMark = 30000;
	auto diskMan = std::shared_ptr<ArchiveSpaceManager>(new ArchiveSpaceManager(ArchiveSpaceManagerProps(lowerWaterMark, upperWaterMark, "./data/archive", 1)));
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
	auto pinId = diskMan->addOutputPin(metadata);
	auto sink = std::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	diskMan->setNext(sink);

	BOOST_TEST(diskMan->init());
	BOOST_TEST(sink->init());

	diskMan->step();
	BOOST_ASSERT(diskMan->finalArchiveSpace < lowerWaterMark);

	createDiskFiles();
	diskMan->step();

	BOOST_ASSERT(diskMan->finalArchiveSpace < lowerWaterMark);
}

BOOST_AUTO_TEST_CASE(getSetProps)
{
	createDiskFiles();
	std::vector<std::string> folder = { "./data/archive" };
	Test_Utils::FileCleaner f(folder);
	uint32_t lowerWaterMark = 25000;
	uint32_t newLowerWaterMark = 15000;
	uint32_t upperWaterMark = 30000;
	auto diskMan = std::shared_ptr<ArchiveSpaceManager>(new ArchiveSpaceManager(ArchiveSpaceManagerProps(lowerWaterMark, upperWaterMark, "./data/archive", 1)));
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
	auto pinId = diskMan->addOutputPin(metadata);
	auto sink = std::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	diskMan->setNext(sink);

	BOOST_TEST(diskMan->init());
	BOOST_TEST(sink->init());

	diskMan->step();
	BOOST_ASSERT(diskMan->finalArchiveSpace < lowerWaterMark);

	auto currentProps = diskMan->getProps();
	currentProps.lowerWaterMark = 15000;
	auto newValue = ArchiveSpaceManagerProps(currentProps);
	diskMan->setProps(newValue);
	diskMan->step();

	createDiskFiles();
	diskMan->step();
	BOOST_ASSERT(diskMan->finalArchiveSpace < newLowerWaterMark);
}

BOOST_AUTO_TEST_CASE(profile, *boost::unit_test::disabled())
{
	uint32_t lowerwatermark = 750000;
	uint32_t upperwatermark = 800000;
	int samplerate = 15;
	auto diskman = std::shared_ptr<ArchiveSpaceManager>(new ArchiveSpaceManager(ArchiveSpaceManagerProps(lowerwatermark, upperwatermark, "c:/users/vinayak/desktop/work/redbull 3.0", samplerate)));
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
	auto pinid = diskman->addOutputPin(metadata);
	auto sink = std::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	diskman->setNext(sink);
	BOOST_TEST(diskman->init());
	BOOST_TEST(sink->init());

	diskman->step();
	BOOST_ASSERT(diskman->finalArchiveSpace < lowerwatermark);
}


BOOST_AUTO_TEST_SUITE_END()