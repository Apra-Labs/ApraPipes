#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include "ExternalSourceModule.h"
#include "ExternalSinkModule.h"
#include "ArchiveSpaceManager.h"
#include "Logger.h"
#include "AIPExceptions.h"
#include "test_utils.h"
#include "Module.h"
#include "PipeLine.h"

BOOST_AUTO_TEST_SUITE(archivespacemanager_tests)

void createDiskFiles ()
{
	boost::filesystem::create_directories("data//archive//Cam1//2022//10//01");
	boost::filesystem::create_directories("data//archive//Cam2//2022//10//01");
	boost::filesystem::create_directories("data//archive//Cam3//2022//10//01");
	boost::filesystem::ofstream a("data//archive//Cam1//2022//10//01//file0.txt");
	for (int i = 0; i < 10023; i++)
	{
		a << "a";
	}
	a.close();
	boost::filesystem::ofstream b("data//archive//Cam2//2022//10//01//file0.txt");
	for (int i = 0; i < 10023; i++)
	{
		b << "b";
	}
	b.close();
	boost::filesystem::ofstream c("data//archive//Cam3//2022//10//01//file0.txt");
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
	auto source = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
	auto pinId = source->addOutputPin(metadata);
	auto diskMan = boost::shared_ptr<ArchiveSpaceManager>(new ArchiveSpaceManager(ArchiveSpaceManagerProps(lowerWaterMark, upperWaterMark, "C://Users//Vinayak//Workspace_ApraPipes//ApraPipes//data//archive",1)));
	source->setNext(diskMan);
	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	diskMan->setNext(sink);

	BOOST_TEST(source->init());
	BOOST_TEST(diskMan->init());
	BOOST_TEST(sink->init());
	auto frame = source->makeFrame(1023, pinId);
	frame_container frames;
	frames.insert(make_pair(pinId, frame));

	source->send(frames);
	diskMan->step();
	BOOST_ASSERT(diskMan->finalArchiveSpace < lowerWaterMark);

}

BOOST_AUTO_TEST_CASE(create_files)
{
	createDiskFiles();
	std::vector<std::string> folder = {"./data/archive"};
	Test_Utils::FileCleaner f(folder);
	uint32_t lowerWaterMark = 25000;
	uint32_t upperWaterMark = 30000;
	auto source = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
	auto pinId = source->addOutputPin(metadata);
	auto diskMan = boost::shared_ptr<ArchiveSpaceManager>(new ArchiveSpaceManager(ArchiveSpaceManagerProps(lowerWaterMark, upperWaterMark, "C://Users//Vinayak//Workspace_ApraPipes//ApraPipes//data//archive",1)));
	source->setNext(diskMan);
	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	diskMan->setNext(sink);

	BOOST_TEST(source->init());
	BOOST_TEST(diskMan->init());
	BOOST_TEST(sink->init());

	auto frame = source->makeFrame(1023, pinId);
	frame_container frames;
	frames.insert(make_pair(pinId, frame));

	source->send(frames);
	diskMan->step();
	BOOST_ASSERT(diskMan->finalArchiveSpace < lowerWaterMark);

	createDiskFiles();
	source->send(frames);
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
	auto source = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
	auto pinId = source->addOutputPin(metadata);
	auto diskMan = boost::shared_ptr<ArchiveSpaceManager>(new ArchiveSpaceManager(ArchiveSpaceManagerProps(lowerWaterMark, upperWaterMark, "C://Users//Vinayak//Workspace_ApraPipes//ApraPipes//data//archive",1)));
	source->setNext(diskMan);
	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	diskMan->setNext(sink);

	BOOST_TEST(source->init());
	BOOST_TEST(diskMan->init());
	BOOST_TEST(sink->init());

	auto frame = source->makeFrame(1023, pinId);
	frame_container frames;
	frames.insert(make_pair(pinId, frame));

	source->send(frames);
	diskMan->step();
	BOOST_ASSERT(diskMan->finalArchiveSpace < lowerWaterMark);

	auto currentProps = diskMan->getProps();
	currentProps.lowerWaterMark = 15000;
	auto newValue = ArchiveSpaceManagerProps(currentProps);
	diskMan->setProps(newValue);
	diskMan->step();

	createDiskFiles();

	source->send(frames);
	diskMan->step();
	BOOST_ASSERT(diskMan->finalArchiveSpace < newLowerWaterMark);
}

BOOST_AUTO_TEST_CASE(profile,*boost::unit_test::disabled())
{
	uint32_t lowerwatermark = 750000;
	uint32_t upperwatermark = 800000;
	int samplerate = 15;
	auto source = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
	auto pinid = source->addOutputPin(metadata);
	auto diskman = boost::shared_ptr<ArchiveSpaceManager>(new ArchiveSpaceManager(ArchiveSpaceManagerProps(lowerwatermark, upperwatermark, "c://users//vinayak//desktop//work//redbull 3.0", samplerate)));

	source->setNext(diskman);
	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	diskman->setNext(sink);

	BOOST_TEST(source->init());
	BOOST_TEST(diskman->init());
	BOOST_TEST(sink->init());
	auto frame = source->makeFrame(1023, pinid);
	frame_container frames;
	frames.insert(make_pair(pinid, frame));

	source->send(frames);
	diskman->step();
	BOOST_ASSERT(diskman->finalArchiveSpace < lowerwatermark);
}


BOOST_AUTO_TEST_SUITE_END()