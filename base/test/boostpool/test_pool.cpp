/* Copyright (C) 2000, 2001 Stephen Cleary
 * Copyright (C) 2011 Kwan Ting Chan
 * 
 * Use, modification and distribution is subject to the 
 * Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or http://www.boost.org/LICENSE_1_0.txt)
 */

#include <boost/test/unit_test.hpp>

#include "random_shuffle.hpp"
#include "ApraPool.h"
#include <boost/pool/pool.hpp>
#include <boost/detail/lightweight_test.hpp>

#include <algorithm>
#include <deque>
#include <list>
#include <set>
#include <stdexcept>
#include <vector>

#include <cstdlib>
#include <ctime>

// Each "tester" object below checks into and out of the "cdtor_checker",
//  which will check for any problems related to the construction/destruction of
//  "tester" objects.
class cdtor_checker
{
private:
    // Each constructed object registers its "this" pointer into "objs"
    std::set<void*> objs;

public:
    // True iff all objects that have checked in have checked out
    bool ok() const { return objs.empty(); }

    ~cdtor_checker()
    {
        BOOST_TEST(ok());
    }

    void check_in(void * const This)
    {
        BOOST_TEST(objs.find(This) == objs.end());
        objs.insert(This);
    }

    void check_out(void * const This)
    {
        BOOST_TEST(objs.find(This) != objs.end());
        objs.erase(This);
    }
};
static cdtor_checker mem;

struct tester
{
    tester(bool throw_except = false)
    {
        if(throw_except)
        {
            throw std::logic_error("Deliberate constructor exception");
        }

        mem.check_in(this);
    }

    tester(const tester &)
    {
        mem.check_in(this);
    }

    ~tester()
    {
        mem.check_out(this);
    }
};

// This is a wrapper around a UserAllocator. It just registers alloc/dealloc
//  to/from the system memory. It's used to make sure pool's are allocating
//  and deallocating system memory properly.
// Do NOT use this class with static or singleton pools.
template <typename UserAllocator>
struct TrackAlloc
{
    typedef typename UserAllocator::size_type size_type;
    typedef typename UserAllocator::difference_type difference_type;

    static std::set<char *> allocated_blocks;

    static char * malloc(const size_type bytes)
    {
        char * const ret = UserAllocator::malloc(bytes);		
        allocated_blocks.insert(ret);
        return ret;
    }

    static void free(char * const block)
    {
        BOOST_TEST(allocated_blocks.find(block) != allocated_blocks.end());
        allocated_blocks.erase(block);		
        UserAllocator::free(block);
    }

    static bool ok()
    {
        return allocated_blocks.empty();
    }
};
template <typename UserAllocator>
std::set<char *> TrackAlloc<UserAllocator>::allocated_blocks;

typedef TrackAlloc<boost::default_user_allocator_new_delete> track_alloc;

void test_mem_usage()
{    
	typedef ApraPool<track_alloc> pool_type;

    {
        // Constructor should do nothing; no memory allocation
        pool_type pool(sizeof(int));
		BOOST_TEST(track_alloc::ok());
        BOOST_TEST(!pool.release_memory());
        BOOST_TEST(!pool.purge_memory());

        // Should allocate from system
        pool.ordered_free(pool.ordered_malloc(1), 1);
        BOOST_TEST(!track_alloc::ok());

        // Ask pool to give up memory it's not using; this should succeed
        BOOST_TEST(pool.release_memory());
        BOOST_TEST(track_alloc::ok());

        // Should allocate from system again
        pool.ordered_malloc(1); // loses the pointer to the returned chunk (*A*)

        // Ask pool to give up memory it's not using; this should fail
        BOOST_TEST(!pool.release_memory());

        // Force pool to give up memory it's not using; this should succeed
        // This will clean up the memory leak from (*A*)
        BOOST_TEST(pool.purge_memory());
        BOOST_TEST(track_alloc::ok());

        // Should allocate from system again
        pool.ordered_malloc(1); // loses the pointer to the returned chunk (*B*)

        // pool's destructor should purge the memory
        //  This will clean up the memory leak from (*B*)
    }

    BOOST_TEST(track_alloc::ok());
}

BOOST_AUTO_TEST_CASE(test_pool)
{
    std::srand(static_cast<unsigned>(std::time(0)));

    test_mem_usage();
  
    boost::report_errors();
}
