/* Copyright (C) 2011 John Maddock
* 
* Use, modification and distribution is subject to the 
* Boost Software License, Version 1.0. (See accompanying
* file LICENSE_1_0.txt or http://www.boost.org/LICENSE_1_0.txt)
*/

// Test of bug #2696 (https://svn.boost.org/trac/boost/ticket/2696)
#include <boost/test/unit_test.hpp>
#include <boost/integer/common_factor_ct.hpp>
#include "ApraPool.h"
#include <boost/detail/lightweight_test.hpp>

struct limited_allocator_new_delete_2696
{
    
   typedef std::ptrdiff_t difference_type; 

   static char * malloc BOOST_PREVENT_MACRO_SUBSTITUTION(const size_t bytes)
   { 

      static const unsigned max_size = sizeof(void*) * 40;

	  if (bytes > max_size)
	  {
		  return 0;
	  }
      return new (std::nothrow) char[bytes];
   }
   static void free BOOST_PREVENT_MACRO_SUBSTITUTION(char * const block)
   { 
      delete [] block;
   }
};

BOOST_AUTO_TEST_CASE(test_pool_bug_2696)
{
   static const unsigned alloc_size = sizeof(void*);
   ApraPool<limited_allocator_new_delete_2696> p1(alloc_size, 10, 40);
   for(int i = 1; i <= 40; ++i)
      BOOST_TEST((p1.ordered_malloc)(i));
   BOOST_TEST(p1.ordered_malloc(42) == 0);
   //
   // If the largest block is 40, and we start with 10, we get 10+20+40 elements before
   // we actually run out of memory:
   //
   ApraPool<limited_allocator_new_delete_2696> p2(alloc_size, 10, 40);
   for(int i = 1; i <= 70; ++i)
      BOOST_TEST((p2.ordered_malloc)(1));
   ApraPool<limited_allocator_new_delete_2696> p2b(alloc_size, 10, 40);
   for(int i = 1; i <= 100; ++i)
      BOOST_TEST((p2b.ordered_malloc)(1));
   //
   // Try again with no explicit upper limit:
   //
   ApraPool<limited_allocator_new_delete_2696> p3(alloc_size);
   for(int i = 1; i <= 40; ++i)
      BOOST_TEST((p3.ordered_malloc)(i));
   BOOST_TEST(p3.ordered_malloc(42) == 0);
   ApraPool<limited_allocator_new_delete_2696> p4(alloc_size, 10);
   for(int i = 1; i <= 100; ++i)
      BOOST_TEST((p4.ordered_malloc)(1));
   ApraPool<limited_allocator_new_delete_2696> p5(alloc_size, 10);
   for(int i = 1; i <= 100; ++i)
      BOOST_TEST((p5.ordered_malloc)(1));
   boost::report_errors();
}
