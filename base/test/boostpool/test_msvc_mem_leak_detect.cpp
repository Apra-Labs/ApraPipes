/* Copyright (C) 2011 Kwan Ting Chan
 * 
 * Use, modification and distribution is subject to the 
 * Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or http://www.boost.org/LICENSE_1_0.txt)
 */

// Test of bug #4346 (https://svn.boost.org/trac/boost/ticket/4346)

#ifdef _MSC_VER
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif
#include <boost/test/unit_test.hpp>
#include <boost/pool/pool.hpp>
#include "ApraPool.h"

#include <vector>

struct Foo {};

BOOST_AUTO_TEST_CASE(test_pool_msvc_mem_leak_detect)
{
    {
        ApraPool<boost::default_user_allocator_new_delete> p(sizeof(int));
        (p.ordered_malloc)(1);
    }

    
}
