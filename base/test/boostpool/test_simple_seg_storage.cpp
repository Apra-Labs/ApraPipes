/* Copyright (C) 2011 Kwan Ting Chan
 *
 * Use, modification and distribution is subject to the
 * Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or http://www.boost.org/LICENSE_1_0.txt)
 */

#include <boost/test/unit_test.hpp>

#include "test_simple_seg_storage.hpp"
#include "track_allocator.hpp"
#include "random_shuffle.hpp"

#include "ApraPool.h"
#include <boost/assert.hpp>
#include <boost/integer/common_factor_ct.hpp>
#if defined(BOOST_MSVC) && (BOOST_MSVC <= 1600)
#pragma warning(push)
#pragma warning(disable: 4244)
 // ..\..\boost/random/uniform_int_distribution.hpp(171) :
 //   warning C4127: conditional expression is constant
#pragma warning(disable: 4127)
#endif
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#if defined(BOOST_MSVC) && (BOOST_MSVC <= 1600)
#pragma warning(pop)
#endif



#include <algorithm>
#include <functional>
#include <set>
#include <vector>

#include <cstddef>
#include <cstdlib>
#include <ctime>

#ifdef BOOST_MSVC
#pragma warning(disable:4267)
#endif

// "A free list is ordered if repeated calls to malloc() will result in a
//  constantly-increasing sequence of values, as determined by std::less<void*>"
// Return: true if in constantly-increasing order, false otherwise
bool check_is_order(const std::vector<void*>& vs)
{
	if (vs.size() < 2) { return true; }

	void *lower, *higher;
	std::vector<void*>::const_iterator ci = vs.begin();
	lower = *(ci++);
	while (ci != vs.end())
	{
		higher = *(ci++);
		if (!std::less<void*>()(lower, higher)) { return false; }
	}

	return true;
}

// Return: number of chunks malloc'd from store
std::size_t test_is_order(test_simp_seg_store& store, const size_t partition_size)
{
	std::vector<void*> vpv;
	std::size_t nchunk = 0;
	// Pre: !empty()
	while (!store.empty())
	{		
		void* const first = store.get_first();
		void* const pv = store.malloc(partition_size);
		// "Takes the first available chunk from the free list
		//  and returns it"
		BOOST_TEST(first == pv);

		vpv.push_back(pv);
		++nchunk;
	}
	BOOST_TEST(check_is_order(vpv));

	return nchunk;
}

boost::mt19937 gen;

BOOST_AUTO_TEST_CASE(test_simp_seg_storage)
{
	std::srand(static_cast<unsigned>(std::time(0)));
	gen.seed(static_cast<boost::uint32_t>(std::time(0)));

	/* Store::segregate(block, sz, partition_sz, end) */
	std::size_t partition_sz
		= boost::integer::static_lcm<sizeof(void*), sizeof(int)>::value;
	boost::uniform_int<> dist(partition_sz, 10000);
	boost::variate_generator<boost::mt19937&,
		boost::uniform_int<> > die(gen, dist);
	std::size_t block_size = die();
	// Pre: npartition_sz >= sizeof(void*)
	//      npartition_sz = sizeof(void*) * i, for some integer i
	//      nsz >= npartition_sz
	//      block is properly aligned for an array of object of
	//        size npartition_sz and array of void *
	BOOST_ASSERT(partition_sz >= sizeof(void*));
	BOOST_ASSERT(partition_sz % sizeof(void*) == 0);
	BOOST_ASSERT(block_size >= partition_sz);
	{
		char* const pc = track_allocator::malloc(block_size);
		// (Test) Pre: block of memory is valid
		BOOST_ASSERT(pc);
		int endadd = 0;
		test_simp_seg_store tempobj;
		void* const pvret = tempobj.segregate(pc, block_size,
			partition_sz, &endadd);

		// The first chunk "is always equal to block"
		BOOST_TEST(pvret == pc);

		void* cur = tempobj.get_nextof(static_cast<int*>(pvret));
		void* last = pvret;
		std::size_t nchunk = 1;
		while (cur != &endadd)
		{
			++nchunk;

			// Memory of each chunk does not overlap
			// The free list constructed is actually from the given block
			// The "interleaved free list is ordered"
			BOOST_TEST(std::less_equal<void*>()(static_cast<char*>(last)
				+ partition_sz, cur));
			BOOST_TEST(std::less_equal<void*>()(static_cast<char*>(cur)
				+ partition_sz, pc + block_size));

			last = cur;
			cur = tempobj.get_nextof(static_cast<int*>(cur));
		}
		// "The last chunk is set to point to end"
		// "Partitioning into as many partition_sz-sized chunks as possible"
		BOOST_TEST(nchunk == block_size / partition_sz);
	}

	/* t.add_block(block, sz, partition_sz), t.malloc() */
	{
		// Default constructor of simple_segregated_storage do nothing
		test_simp_seg_store tstore;
		// Post: empty()
		BOOST_TEST(tstore.empty());

		char* const pc = track_allocator::malloc(block_size);
		tstore.add_block(pc, block_size, partition_sz);

		// The first chunk "is always equal to block"
		BOOST_TEST(tstore.get_first() == pc);

		// Empty before add_block() => "is ordered after"
		std::size_t nchunk = test_is_order(tstore, partition_sz);
		// "Partitioning into as many partition_sz-sized chunks as possible"
		BOOST_TEST(nchunk == block_size / partition_sz);

		BOOST_ASSERT(partition_sz <= 23);
		test_simp_seg_store tstore2;
		char* const pc2 = track_allocator::malloc(88);
		tstore2.add_block(pc2, 24, partition_sz);
		tstore2.add_block(pc2 + 64, 24, partition_sz);
		tstore2.add_block(pc2 + 32, 24, partition_sz);
		tstore2.add_block(track_allocator::malloc(23), 23, partition_sz);
		std::size_t nchunk_ref = (3 * (24 / partition_sz)) + (23 / partition_sz);
		for (nchunk = 0; !tstore2.empty(); tstore2.malloc(partition_sz), ++nchunk) {}
		// add_block() merges new free list to existing
		BOOST_TEST(nchunk == nchunk_ref);
	}

	/* t.free(chunk) */
	{
		test_simp_seg_store tstore;
		char* const pc = track_allocator::malloc(partition_sz);
		tstore.add_block(pc, partition_sz, partition_sz);
		void* pv = tstore.malloc(partition_sz);
		BOOST_TEST(tstore.empty());
		tstore.free(pv, partition_sz);
	}

	/* t.add_ordered_block(block, sz, partition_sz) */
	{
		{
			char* const pc = track_allocator::malloc(6 * partition_sz);
			std::vector<void*> vpv;
			vpv.push_back(pc);
			vpv.push_back(pc + (2 * partition_sz));
			vpv.push_back(pc + (4 * partition_sz));

			do
			{
				test_simp_seg_store tstore;
				tstore.add_ordered_block(vpv[0], 2 * partition_sz, partition_sz);
				tstore.add_ordered_block(vpv[1], 2 * partition_sz, partition_sz);
				tstore.add_ordered_block(vpv[2], 2 * partition_sz, partition_sz);
				// "Order-preserving"
				test_is_order(tstore, partition_sz);
			} while (std::next_permutation(vpv.begin(), vpv.end()));
		}

		{
			test_simp_seg_store tstore;
			char* const pc = track_allocator::malloc(6 * partition_sz);
			tstore.add_ordered_block(pc, 2 * partition_sz, partition_sz);
			tstore.add_ordered_block(pc + (4 * partition_sz),
				(2 * partition_sz), partition_sz);
			// "Order-preserving"
			test_is_order(tstore, partition_sz);
		}

		{
			test_simp_seg_store tstore;
			char* const pc = track_allocator::malloc(6 * partition_sz);
			tstore.add_ordered_block(pc + (4 * partition_sz),
				(2 * partition_sz), partition_sz);
			tstore.add_ordered_block(pc, 2 * partition_sz, partition_sz);
			// "Order-preserving"
			test_is_order(tstore, partition_sz);
		}
	}

	/* t.ordered_free(chunk) */
	{
		char* const pc = track_allocator::malloc(6 * partition_sz);

		test_simp_seg_store tstore;
		tstore.add_block(pc, 6 * partition_sz, partition_sz);

		std::vector<void*> vpv;
		for (std::size_t i = 0; i < 6; ++i) 
		{
			void* tmptr = tstore.malloc(partition_sz);
			vpv.push_back(tmptr); 
		}
		BOOST_ASSERT(tstore.empty());
		pool_test_random_shuffle(vpv.begin(), vpv.end());

		for (std::size_t i = 0; i < 6; ++i)
		{
			tstore.ordered_free(vpv[i], partition_sz);
		}
		// "Order-preserving"
		test_is_order(tstore, partition_sz);
	}

	/* t.malloc_n(n, partition_sz) */
	{
		{
			char* const pc = track_allocator::malloc(12 * partition_sz);
			test_simp_seg_store tstore;
			tstore.add_ordered_block(pc, 2 * partition_sz, partition_sz);
			tstore.add_ordered_block(pc + (3 * partition_sz),
				3 * partition_sz, partition_sz);
			tstore.add_ordered_block(pc + (7 * partition_sz),
				5 * partition_sz, partition_sz);

			void* pvret = tstore.malloc_n(6, partition_sz);
			BOOST_TEST(pvret == 0);

			pvret = tstore.malloc_n(0, partition_sz);
			// There's no prohibition against asking for zero elements
			BOOST_TEST(pvret == 0);

			pvret = tstore.malloc_n(3, partition_sz);
			// Implicit assumption that contiguous sequence found is the first
			//  available while traversing from the start of the free list
			BOOST_TEST(pvret == pc + (3 * partition_sz));

			pvret = tstore.malloc_n(4, partition_sz);
			BOOST_TEST(pvret == pc + (7 * partition_sz));

			// There should still be two contiguous
			//  and one non-contiguous chunk left
			std::size_t nchunks = 0;
			while (!tstore.empty())
			{
				tstore.malloc(partition_sz);
				++nchunks;
			}
			BOOST_TEST(nchunks == 3);
		}

		{
			char* const pc = track_allocator::malloc(12 * partition_sz);
			test_simp_seg_store tstore;
			tstore.add_ordered_block(pc, 2 * partition_sz, partition_sz);
			tstore.add_ordered_block(pc + (3 * partition_sz),
				3 * partition_sz, partition_sz);
			tstore.add_ordered_block(pc + (7 * partition_sz),
				5 * partition_sz, partition_sz);

			tstore.malloc_n(3, partition_sz);
			// "Order-preserving"
			test_is_order(tstore, partition_sz);
		}
	}

	for (std::set<char*>::iterator itr
		= track_allocator::allocated_blocks.begin();
		itr != track_allocator::allocated_blocks.end();
		++itr)
	{
		delete[] * itr;
	}
	track_allocator::allocated_blocks.clear();

	boost::report_errors();	
}
