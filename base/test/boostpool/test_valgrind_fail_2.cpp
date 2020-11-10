/* Copyright (C) 2011 John Maddock
* 
* Use, modification and distribution is subject to the 
* Boost Software License, Version 1.0. (See accompanying
* file LICENSE_1_0.txt or http://www.boost.org/LICENSE_1_0.txt)
*/

// Test of bug #2656 (https://svn.boost.org/trac/boost/ticket/2656)

#include "ApraPool.h"
#include <boost/test/unit_test.hpp>
#include <boost/pool/pool.hpp>

BOOST_AUTO_TEST_CASE(test_pool_valgrind_fail_2)
{
   ApraPool<boost::default_user_allocator_new_delete> p(sizeof(int));
   int* ptr = static_cast<int*>((p.ordered_malloc)(1));
   *ptr = 0;
   (p.ordered_free)(ptr, 1);
   *ptr = 2; // write to freed memory   
}
