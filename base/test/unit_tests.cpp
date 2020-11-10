#include "stdafx.h"
#include "Utils.h"
#include "Frame.h"
#include "FrameContainerQueue.h"
#include "FrameFactory.h"
#include "Module.h"
#include "PipeLine.h"
#include <boost/test/unit_test.hpp>

// NOTE: TESTS WHICH REQUIRE ANY ENVIRONMENT TO BE PRESENT BEFORE RUNNING ARE NOT UNIT TESTS !!!


BOOST_AUTO_TEST_SUITE(unit_tests)

BOOST_AUTO_TEST_CASE(dummy_test)
{
	
}

BOOST_AUTO_TEST_CASE(frame_factory_test)
{
	boost::shared_ptr<FrameFactory> fact(new FrameFactory);
	auto f1 = fact->create(1023, fact, FrameMetadata::MemType::HOST);//uses 1 chunk
	auto f2 = fact->create(1024, fact, FrameMetadata::MemType::HOST);//uses 1 chunk
	auto f3 = fact->create(1025, fact, FrameMetadata::MemType::HOST);//uses 2 chunks
	auto f4 = fact->create(2047, fact, FrameMetadata::MemType::HOST);//uses 2 chunk
	auto f5 = fact->create(100000, fact, FrameMetadata::MemType::HOST); //uses 98 chunk
}

BOOST_AUTO_TEST_CASE(multiple_que_test)
{
	{
		boost::shared_ptr<FrameContainerQueue> q1 = boost::shared_ptr<FrameContainerQueue>(new FrameContainerQueue(20));
		boost::shared_ptr<FrameFactory> fact(new FrameFactory);

		{
			auto f = fact->create(1023, fact, FrameMetadata::MemType::HOST);//uses 1 chunk
			frame_container frames;
			frames["a"] = f;
			q1->push(frames);
		}

		

		fact.reset();
		q1->clear();
	}

	boost::this_thread::sleep_for(boost::chrono::milliseconds(500));

}

#ifdef APRA_CUDA_ENABLED

BOOST_AUTO_TEST_CASE(frame_factory_test_host_pinned)
{
	boost::shared_ptr<FrameFactory> fact(new FrameFactory);
	auto f1 = fact->create(1023, fact, FrameMetadata::MemType::HOST_PINNED);//uses 1 chunk
	auto f2 = fact->create(1024, fact, FrameMetadata::MemType::HOST_PINNED);//uses 1 chunk
	auto f3 = fact->create(1025, fact, FrameMetadata::MemType::HOST_PINNED);//uses 2 chunks
	auto f4 = fact->create(2047, fact, FrameMetadata::MemType::HOST_PINNED);//uses 2 chunk
	auto f5 = fact->create(100000, fact, FrameMetadata::MemType::HOST_PINNED); //uses 98 chunk
}

BOOST_AUTO_TEST_CASE(frame_factory_test_cuda_device)
{
	boost::shared_ptr<FrameFactory> fact(new FrameFactory);
	auto f1 = fact->create(1023, fact, FrameMetadata::MemType::CUDA_DEVICE);//uses 1 chunk
	auto f2 = fact->create(1024, fact, FrameMetadata::MemType::CUDA_DEVICE);//uses 1 chunk
	auto f3 = fact->create(1025, fact, FrameMetadata::MemType::CUDA_DEVICE);//uses 2 chunks
	auto f4 = fact->create(2047, fact, FrameMetadata::MemType::CUDA_DEVICE);//uses 2 chunk
	auto f5 = fact->create(100000, fact, FrameMetadata::MemType::CUDA_DEVICE); //uses 98 chunk
}

#endif

BOOST_AUTO_TEST_CASE(boost_pool_ordered_malloc_free)
{
	static size_t noOfNew = 0;
	static size_t noOfFree = 0;
	static void* myPtr = NULL;

	struct apra_allocator
	{
		typedef std::size_t size_type;
		typedef std::ptrdiff_t difference_type;

		static char *malloc(const size_type bytes)
		{
			auto ptr = new (std::nothrow) char[bytes];
			myPtr = ptr;
			noOfNew++;
			return ptr;
		}
		static void free(char *const block)
		{
			noOfFree++;
			BOOST_TEST((void*)block == myPtr);
			delete[] block;
		}
	};

	size_t CHUNK_SZ = 1024;
	boost::pool<apra_allocator> p(CHUNK_SZ);

	size_t noOfChunks = 100;
	size_t totalSize = noOfChunks * CHUNK_SZ;

	for (int i = 0; i < 2; i++)
	{
		unsigned char *buffer = (unsigned char *)p.ordered_malloc(noOfChunks);
		p.ordered_free(buffer, noOfChunks);
	}

	unsigned char *buffer = (unsigned char *)p.ordered_malloc(noOfChunks);
	memset(buffer, 1, totalSize);
	{
		size_t noOfErrors = 0;
		for (auto i = 0; i < totalSize; i++)
		{

			if (buffer[i] != 1)
			{
				noOfErrors += 1;
			}


		}
		BOOST_TEST(noOfErrors == 0);
	}
	// calling free for 2nd half of chunks
	p.ordered_free(buffer + (totalSize / 2), noOfChunks / 2);

	{
		size_t noOfErrors = 0;
		for (auto i = 0; i < totalSize / 2; i++)
		{
			if (buffer[i] != 1)
			{
				noOfErrors += 1;
			}


		}
		BOOST_TEST(noOfErrors == 0);
	}

	// asking for half of chunks
	unsigned char *buffer2 = (unsigned char *)p.ordered_malloc(noOfChunks / 2);
	{
		size_t noOfErrors = 0;
		for (auto i = 0; i < totalSize/2; i++)
		{
			if (i < totalSize / 2)
			{
				if (buffer[i] != 1)
				{
					noOfErrors += 1;
				}
			}
			else
			{
				if (buffer[i] != 2)
				{
					noOfErrors += 1;
				}
			}

		}
		BOOST_TEST(noOfErrors == 0);
	}

	memset(buffer2, 2, totalSize / 2);
	{
		size_t noOfErrors = 0;
		for (auto i = 0; i < totalSize; i++)
		{
			if (i < totalSize / 2)
			{
				if (buffer[i] != 1)
				{
					noOfErrors += 1;
				}
			}
			else
			{
				if (buffer[i] != 2)
				{
					noOfErrors += 1;
				}
			}

		}
		BOOST_TEST(noOfErrors == 0);
	}
	// calling free on the buffer2
	p.ordered_free(buffer2, noOfChunks / 2);

	{
		size_t noOfErrors = 0;
		for (auto i = 0; i < totalSize / 2; i++)
		{
			if (i < totalSize / 2)
			{
				if (buffer[i] != 1)
				{
					noOfErrors += 1;
				}
			}
			else
			{
				if (buffer[i] != 2)
				{
					noOfErrors += 1;
				}
			}

		}
		BOOST_TEST(noOfErrors == 0);
	}

	// calling free for 1st half of chunks
	p.ordered_free(buffer, noOfChunks / 2);

	{
		size_t noOfErrors = 0;
		for (auto i = 0; i < totalSize; i++)
		{
			if (i < totalSize / 2)
			{
				if (buffer[i] != 1)
				{
					noOfErrors += 1;
				}
			}
			else
			{
				if (buffer[i] != 2)
				{
					noOfErrors += 1;
				}
			}

		}
		BOOST_TEST(noOfErrors != 0);
	}

	// asking for chunks again and freeing
	unsigned char *buffer3 = (unsigned char *)p.ordered_malloc(noOfChunks);
	memset(buffer3, 3, totalSize);
	{
		size_t noOfErrors = 0;
		for (auto i = 0; i < totalSize; i++)
		{

			if (buffer3[i] != 3)
			{
				noOfErrors += 1;
			}


		}
		BOOST_TEST(noOfErrors == 0);            
	}

	p.ordered_free(buffer3, noOfChunks);

	{
		size_t noOfErrors = 0;
		for (auto i = 0; i < totalSize; i++)
		{
			if (buffer3[i] != 3)
			{
				noOfErrors += 1;
			}


		}
		BOOST_TEST(noOfErrors != 0);
	}

	unsigned char *buffer4 = (unsigned char *)p.ordered_malloc(noOfChunks);
	p.ordered_free(buffer4, noOfChunks);

	unsigned char *buffer5 = (unsigned char *)p.ordered_malloc(noOfChunks / 2);
	unsigned char *buffer6 = (unsigned char *)p.ordered_malloc(noOfChunks / 2);
	p.ordered_free(buffer5, noOfChunks / 2);
	p.ordered_free(buffer6, noOfChunks / 2);

	unsigned char *buffer7 = (unsigned char *)p.ordered_malloc(noOfChunks);
	p.ordered_free(buffer7, noOfChunks);

	void* secondptr = (void*)(buffer + (totalSize / 2));
	BOOST_TEST((void*)buffer == myPtr);
	BOOST_TEST((void*)buffer2 == secondptr);
	BOOST_TEST((void*)buffer3 == myPtr);
	BOOST_TEST((void*)buffer4 == myPtr);
	BOOST_TEST((void*)buffer5 == myPtr);
	BOOST_TEST((void*)buffer6 == secondptr);
	BOOST_TEST((void*)buffer7 == myPtr);

	p.release_memory();

	BOOST_TEST(noOfNew == 1);
	BOOST_TEST(noOfFree == 1);
}

BOOST_AUTO_TEST_CASE(frame_factory_resize_test)
{
	boost::shared_ptr<FrameFactory> fact(new FrameFactory);
	{
		auto buffer = fact->createBuffer(5000, fact, FrameMetadata::MemType::HOST);
		BOOST_TEST(buffer->size() == 5000);

		auto frame = fact->create(buffer, 2000, fact, FrameMetadata::MemType::HOST);
		BOOST_TEST(frame->size() == 2000);
	}

	{
		auto buffer = fact->createBuffer(5000, fact, FrameMetadata::MemType::HOST);
		BOOST_TEST(buffer->size() == 5000);

		auto frame = fact->create(buffer, 5000, fact, FrameMetadata::MemType::HOST);
		BOOST_TEST(frame->size() == 5000);
	}
}

BOOST_AUTO_TEST_CASE(base64_encoding_test)
{
	std::string authString1 = "admin:";
	std::string authString2 = "admin:passwd";

	std::string res1 = Utils::base64_encode(reinterpret_cast<const unsigned char*>(authString1.c_str()), authString1.length());
	std::string res2 = Utils::base64_encode(reinterpret_cast<const unsigned char*>(authString2.c_str()), authString2.length());
	BOOST_TEST(res1.compare("YWRtaW46") == 0);
	BOOST_TEST(res2.compare("YWRtaW46cGFzc3dk") == 0);
}

void producer_consumer(bool isProducer, boost::shared_ptr<FrameFactory> fact, FrameContainerQueue& q, int iterations)
{
	for (int i = 0; i < iterations; i++)
	{
		if (isProducer)
		{
			auto f1 = fact->create(1023, fact, FrameMetadata::MemType::HOST);//uses 1 chunk
			frame_container frames;
			frames.insert(std::make_pair("p1", f1));
			q.push(frames);
		}
		else {
			auto f1 = q.pop();
		}
		//		if (i % 1000 == 999)
		//			cout << (isProducer ? "Producer" : "Consumer") << " finished " << std::to_string(i) << " iterations " << endl;
	}
}

BOOST_AUTO_TEST_CASE(two_threaed_framefactory_test)
{
	FrameContainerQueue q(20);
	boost::shared_ptr<FrameFactory> fact(new FrameFactory);
	std::thread t1(producer_consumer, true, fact, std::ref(q), 100000);
	std::thread t2(producer_consumer, false, fact, std::ref(q), 100000);

	t1.join();
	t2.join();

}

BOOST_AUTO_TEST_SUITE_END()