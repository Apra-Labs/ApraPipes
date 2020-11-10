#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>

#include "ExternalSinkModule.h"
#include "FileReaderModule.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "AIPExceptions.h"
#include "test_utils.h"
#include "PipeLine.h"
#include "StatSink.h"

BOOST_AUTO_TEST_SUITE(filereadermodule_tests)

BOOST_AUTO_TEST_CASE(basic)
{
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/filenamestrategydata/?.txt")));
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::GENERAL));
	auto pinId = fileReader->addOutputPin(metadata);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	fileReader->setNext(sink);
		
	BOOST_TEST(fileReader->init());
	BOOST_TEST(sink->init());

	fileReader->play(true);

	for (auto i = 0; i < 10; i++)
	{
		fileReader->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(pinId) != frames.end()));
		auto frame = frames[pinId];
		auto buffer = (unsigned char*)frame->data();

		BOOST_TEST(frame->fIndex2 == i);
		BOOST_TEST(buffer[0] == 48+i);
		BOOST_TEST(buffer[1] == 48+i);
	}

	fileReader->play(false);

	for (auto i = 0; i < 10; i++)
	{
		fileReader->step();
		auto frames = sink->try_pop();
		BOOST_TEST((frames.find(pinId) == frames.end()));		
	}

	fileReader->play(true);
	fileReader->jump(5);
	
	for (auto i = 5; i < 10; i++)
	{
		fileReader->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(pinId) != frames.end()));
		auto frame = frames[pinId];
		auto buffer = (unsigned char*)frame->data();

		BOOST_TEST(frame->fIndex2 == i);
		BOOST_TEST(buffer[0] == 48 + i);
		BOOST_TEST(buffer[1] == 48 + i);
	}

	fileReader->jump(2);
	for (auto i = 2; i < 10; i++)
	{
		fileReader->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(pinId) != frames.end()));
		auto frame = frames[pinId];
		auto buffer = (unsigned char*)frame->data();

		BOOST_TEST(frame->fIndex2 == i);
		BOOST_TEST(buffer[0] == 48 + i);
		BOOST_TEST(buffer[1] == 48 + i);
	}


	fileReader->term();
	sink->term();
}

BOOST_AUTO_TEST_CASE(relay)
{
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/filenamestrategydata/?.txt")));
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::GENERAL));
	auto pinId = fileReader->addOutputPin(metadata);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	fileReader->setNext(sink, false);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(sink->init());

	fileReader->play(true);

	for (auto i = 0; i < 10; i++)
	{
		fileReader->step();
		auto frames = sink->try_pop();
		BOOST_TEST(frames.size() == 0);
	}

	fileReader->relay(sink, true);
	for (auto i = 0; i < 10; i++)
	{
		fileReader->step();
		auto frames = sink->try_pop();
		BOOST_TEST(frames.size() == 1);
	}

	fileReader->relay(sink, false);
	for (auto i = 0; i < 10; i++)
	{
		fileReader->step();
		auto frames = sink->try_pop();
		BOOST_TEST(frames.size() == 0);
	}
	
}

BOOST_AUTO_TEST_CASE(pipeline_relay)
{
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/filenamestrategydata/?.txt")));
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::GENERAL));
	auto pinId = fileReader->addOutputPin(metadata);

	bool relay = false;
	auto sink = boost::shared_ptr<Module>(new StatSink());	
	fileReader->setNext(sink, relay);

	PipeLine p("test");
	p.appendModule(fileReader);
	p.init();
	p.run_all_threaded();

	for (auto i = 0; i < 10; i++)
	{
		boost::this_thread::sleep_for(boost::chrono::milliseconds(100));  // giving time to call step 
		relay = !relay;
		fileReader->relay(sink, relay);
	}

	p.stop();
	p.term();
	p.wait_for_all();

}

BOOST_AUTO_TEST_CASE(configpipeline, *boost::unit_test::disabled())
{
	std::string rootDir = "RecordingFolder/5e9ee85ba832470bc8331109"; // point to folder of jpegs
	auto fileReaderProps = FileReaderModuleProps(rootDir, 0, -1, 4 * 1024 * 1024);
	fileReaderProps.fps = 10;
	fileReaderProps.readLoop = false;
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));
	auto encodedImageMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	fileReader->addOutputPin(encodedImageMetadata); 

	class FileSinkModuleProps : public ModuleProps
	{
	public:
		FileSinkModuleProps() : ModuleProps() {}
	};

	class FileSinkModule : public Module
	{
	public:
		FileSinkModule(FileSinkModuleProps props = FileSinkModuleProps()) : Module(SINK, "ExternalSinkModule", props)
		{

		}

		virtual ~FileSinkModule() {}

		frame_container pop()
		{
			return Module::pop();
		}

		frame_container try_pop()
		{
			return Module::try_pop();
		}
	protected:
		bool validateInputPins()
		{
			return true;
		}

		bool process(frame_container& frames)
		{
			for (auto &it : frames)
			{
				std::cout << it.second->fIndex2 << "<>" << it.second->size() << std::endl;
			}

			return true;
		}
	};

	auto sink = boost::shared_ptr<FileSinkModule>(new FileSinkModule());
	fileReader->setNext(sink);

	auto p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(fileReader);

	p->init();	
	p->run_all_threaded_withpause();

	std::cout << "step" << std::endl;
	p->step();

	boost::this_thread::sleep_for(boost::chrono::milliseconds(1000));

	std::cout << "play" << std::endl;
	p->play();

	boost::this_thread::sleep_for(boost::chrono::milliseconds(1000));
	
	p->pause();
	std::cout << "pause" << std::endl;

	std::vector<std::string> files;
	files.push_back("1587472475809.jpg");
	files.push_back("1587472476328.jpg");
	files.push_back("1587472476829.jpg");
	files.push_back("1587472477161.jpg");
	files.push_back("1587472479293.jpg");
	files.push_back("1587472481846.jpg");
	files.push_back("1587472483643.jpg");
	files.push_back("1587472485825.jpg");
	files.push_back("1587472487810.jpg");
	auto listProps = fileReader->getProps();
	listProps.strFullFileNameWithPattern = rootDir;
	listProps.files = files;
	fileReader->setProps(listProps);
	boost::this_thread::sleep_for(boost::chrono::milliseconds(250));
	std::cout << "step files" << std::endl;
	
	p->step();

	boost::this_thread::sleep_for(boost::chrono::milliseconds(500));

	std::cout << "play" << std::endl;
	p->play();

	boost::this_thread::sleep_for(boost::chrono::milliseconds(1000));

	p->pause();
	std::cout << "pause" << std::endl;

	auto dirProps = fileReader->getProps();
	dirProps.strFullFileNameWithPattern = rootDir;
	dirProps.files.clear();
	fileReader->setProps(dirProps);
	boost::this_thread::sleep_for(boost::chrono::milliseconds(250));
	std::cout << "step folder" << std::endl;
	p->step();

	boost::this_thread::sleep_for(boost::chrono::milliseconds(1000));
	

	p->stop();
	p->term();

	p->wait_for_all();

}

BOOST_AUTO_TEST_CASE(propschange)
{	
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/filenamestrategydata/?.txt")));
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::GENERAL));
	auto pinId = fileReader->addOutputPin(metadata);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	fileReader->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(sink->init());

	fileReader->play(true);

	for (auto i = 0; i < 30; i++)
	{
		fileReader->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(pinId) != frames.end()));
		auto frame = frames[pinId];
		auto buffer = (unsigned char*)frame->data();

		auto index = i % 10;

		BOOST_TEST(frame->fIndex2 == (i % 10));
		BOOST_TEST(buffer[0] == 48 + index);
		BOOST_TEST(buffer[1] == 48 + index);
	}

	auto props = fileReader->getProps();
	props.strFullFileNameWithPattern = "./data/filenamestrategydata";
	fileReader->setProps(props);
	fileReader->step();
	{
		auto frames = sink->pop();
		auto frame = frames.begin()->second;
		BOOST_TEST(frame->isEOS());

		BOOST_TEST(sink->pop().size() == 1);
	}

	for (auto i = 1; i < 30; i++)
	{
		fileReader->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(pinId) != frames.end()));
		auto frame = frames[pinId];
		auto buffer = (unsigned char*)frame->data();

		auto index = i % 10;

		BOOST_TEST(frame->fIndex2 == (i % 10));
		BOOST_TEST(buffer[0] == 48 + index);
		BOOST_TEST(buffer[1] == 48 + index);
	}

	FileReaderModuleProps listProps;
	listProps.strFullFileNameWithPattern = "./data/filenamestrategydata";
	for (auto i = 0; i < 5; i++)
	{
		listProps.files.push_back(std::to_string(i*2) + ".txt");
	}
	fileReader->setProps(listProps);
	fileReader->step();
	{
		auto frames = sink->pop();
		auto frame = frames.begin()->second;
		BOOST_TEST(frame->isEOS());
		BOOST_TEST(sink->pop().size() == 1);
	}

	for (auto i = 1; i < 30; i++)
	{
		fileReader->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(pinId) != frames.end()));
		auto frame = frames[pinId];
		auto buffer = (unsigned char*)frame->data();

		auto index = (i % 5)*2;

		BOOST_TEST(frame->fIndex2 == (i % 5));
		BOOST_TEST(buffer[0] == 48 + index);
		BOOST_TEST(buffer[1] == 48 + index);
	}

	
	auto dirProps = FileReaderModuleProps("./data/filenamestrategydata");
	fileReader->setProps(dirProps);
	fileReader->step();
	{
		auto frames = sink->pop();
		auto frame = frames.begin()->second;
		BOOST_TEST(frame->isEOS());
		BOOST_TEST(sink->pop().size() == 1);
	}

	for (auto i = 1; i < 30; i++)
	{
		fileReader->step();
		auto frames = sink->pop();
		BOOST_TEST((frames.find(pinId) != frames.end()));
		auto frame = frames[pinId];
		auto buffer = (unsigned char*)frame->data();

		auto index = i % 10;

		BOOST_TEST(frame->fIndex2 == (i % 10));
		BOOST_TEST(buffer[0] == 48 + index);
		BOOST_TEST(buffer[1] == 48 + index);
	}


	fileReader->term();
	sink->term();
}

BOOST_AUTO_TEST_SUITE_END()