#include "stdafx.h"
#include <boost/test/unit_test.hpp>

#include "FilenameStrategy.h"
#include "AIPExceptions.h"

BOOST_AUTO_TEST_SUITE(filenamestrategy_tests)

BOOST_AUTO_TEST_CASE(boostdirectorystrategy)
{
	uint64_t index = 0;
	std::string dirPath = "./data/filenamestrategydata";
	{
		auto strategy = FilenameStrategy::getStrategy(dirPath, 0, -1, true);
		strategy->Connect();
		strategy->play(true);

		for (auto i = 0; i < 30; i++)
		{
			auto filenameIndex = i % 10;
			BOOST_TEST(strategy->GetFileNameToUse(true, index) == dirPath+ "/" + std::to_string(filenameIndex) + ".txt");
			BOOST_TEST(index == (i % 10));
		}
				
		strategy->SetReadLoop(false);
		for (auto i = 0; i < 5; i++)
		{
			BOOST_TEST(strategy->GetFileNameToUse(true, index).empty());
		}

		strategy->Disconnect();
	}
	
	{
		auto strategy = FilenameStrategy::getStrategy(dirPath, 0, -1, true);
		strategy->Connect();
		strategy->play(true);

		for (auto i = 0; i < 30; i++)
		{
			auto filenameIndex = i % 10;
			BOOST_TEST(strategy->GetFileNameToUse(true, index) == dirPath + "/" + std::to_string(filenameIndex) + ".txt");
			BOOST_TEST(index == (i % 10));
		}

		strategy->Disconnect();
		for (auto i = 0; i < 5; i++)
		{
			BOOST_TEST(strategy->GetFileNameToUse(true, index).empty());
		}		
	}

	{
		auto strategy = FilenameStrategy::getStrategy(dirPath, 5, 8, true);
		strategy->Connect();
		strategy->play(true);

		for (auto i = 0; i < 30; i++)
		{
			auto filenameIndex = (i % 4) + 5;
			BOOST_TEST(strategy->GetFileNameToUse(true, index) == dirPath + "/" + std::to_string(filenameIndex) + ".txt");
			BOOST_TEST(index == (i % 4) + 5);
		}

		strategy->Disconnect();
	}
		
	{
		auto emptyDirPath = "./data/filenamestrategydata/5";
		if (!boost::filesystem::exists(emptyDirPath))
		{
			boost::filesystem::create_directory(emptyDirPath);
		}

		auto strategy = FilenameStrategy::getStrategy(emptyDirPath, 0, -1, true);

		try
		{
			strategy->Connect();
			BOOST_TEST(false);
		}
		catch (AIP_Exception& exception)
		{
			BOOST_TEST(exception.getCode() == AIP_NOTFOUND);
		}
		catch (...)
		{
			BOOST_TEST(false);
		}

		strategy->Disconnect();
	}

	{
		auto strategy = FilenameStrategy::getStrategy(dirPath, 20, -1, true);
		
		try
		{
			strategy->Connect();
			BOOST_TEST(false);
		}
		catch (AIP_Exception& exception)
		{
			BOOST_TEST(exception.getCode() == AIP_PARAM_OUTOFRANGE);
		}
		catch (...)
		{
			BOOST_TEST(false);
		}

		strategy->Disconnect();
	}
		
	{
		// play pause test
		auto strategy = FilenameStrategy::getStrategy(dirPath, 0, -1, true);
		strategy->Connect();
		strategy->play(true);

		for (auto i = 0; i < 30; i++)
		{
			auto filenameIndex = i % 10;
			BOOST_TEST(strategy->GetFileNameToUse(true, index) == dirPath + "/" + std::to_string(filenameIndex) + ".txt");
			BOOST_TEST(index == (i % 10));
		}

		strategy->play(false);

		for (auto i = 0; i < 30; i++)
		{
			auto filenameIndex = 0;
			BOOST_TEST(strategy->GetFileNameToUse(true, index) == dirPath + "/" + std::to_string(filenameIndex) + ".txt");
			BOOST_TEST(index == 0);
		}

		strategy->play(true);

		for (auto i = 0; i < 30; i++)
		{
			auto filenameIndex = i % 10;
			BOOST_TEST(strategy->GetFileNameToUse(true, index) == dirPath + "/" + std::to_string(filenameIndex) + ".txt");
			BOOST_TEST(index == (i % 10));
		}

		strategy->Disconnect();
	}
}

BOOST_AUTO_TEST_CASE(liststrategy)
{
	uint64_t index = 0;
	std::vector<std::string> files;
	for (auto i = 0; i < 10; i++)
	{
		files.push_back(std::to_string(i*3));
	}

	{
		auto strategy = FilenameStrategy::getStrategy("", 0, -1, true, files);
		strategy->Connect();
		strategy->play(true);

		for (auto i = 0; i < 30; i++)
		{
			auto filenameIndex = i % 10;
			BOOST_TEST(strategy->GetFileNameToUse(true, index) == files[filenameIndex]);
			BOOST_TEST(index == (i % 10));
		}

		strategy->SetReadLoop(false);
		for (auto i = 0; i < 5; i++)
		{
			BOOST_TEST(strategy->GetFileNameToUse(true, index).empty());
		}
		
		strategy->SetReadLoop(true);
		for (auto i = 0; i < 5; i++)
		{
			auto filenameIndex = i % 10;
			BOOST_TEST(strategy->GetFileNameToUse(true, index) == files[filenameIndex]);
			BOOST_TEST(index == (i % 10));
		}

		strategy->jump(2);
		for (auto i = 2; i < 10; i++)
		{
			auto filenameIndex = i % 10;
			BOOST_TEST(strategy->GetFileNameToUse(true, index) == files[filenameIndex]);
			BOOST_TEST(index == (i % 10));
		}

		strategy->jump(5);
		for (auto i = 5; i < 30; i++)
		{
			auto filenameIndex = i % 10;
			BOOST_TEST(strategy->GetFileNameToUse(true, index) == files[filenameIndex]);
			BOOST_TEST(index == (i % 10));
		}

		strategy->Disconnect();
	}
	
}

BOOST_AUTO_TEST_SUITE_END()