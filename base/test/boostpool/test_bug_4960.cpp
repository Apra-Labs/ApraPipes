/* Copyright (C) 2011 Kwan Ting Chan
 * Based from bug report submitted by Xiaohan Wang
 * 
 * Use, modification and distribution is subject to the 
 * Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or http://www.boost.org/LICENSE_1_0.txt)
 */

// Test of bug #4960 (https://svn.boost.org/trac/boost/ticket/4960)
#include <boost/test/unit_test.hpp>

#include "ApraPool.h"
#include <boost/pool/pool.hpp>
#include <vector>
#include <iostream>

BOOST_AUTO_TEST_CASE(test_pool_bug_4960)
{
  int limit = 100;
   ApraPool<boost::default_user_allocator_new_delete> po(4);
   for(int i = 0; i < limit; ++i)
   {
      void* p = po.ordered_malloc(0);
      po.ordered_free(p, 0);
   }  

}
