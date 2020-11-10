/* Copyright (C) 2011 Kwan Ting Chan
 * Based from bug report submitted by Xiaohan Wang
 * 
 * Use, modification and distribution is subject to the 
 * Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or http://www.boost.org/LICENSE_1_0.txt)
 */

// Test of bug #3349 (https://svn.boost.org/trac/boost/ticket/3349)
#include <boost/test/unit_test.hpp>
#include <boost/core/lightweight_test.hpp>
#include <boost/pool/pool.hpp>
#include "ApraPool.h"



BOOST_AUTO_TEST_CASE(test_pool_bug_3349)
{
    ApraPool<boost::default_user_allocator_new_delete> p(256, 4);

    void* pBlock1 = p.ordered_malloc( 1 );
    void* pBlock2 = p.ordered_malloc( 4 );
    (void)pBlock2; // warning suppression

    p.ordered_free( pBlock1, 1 );

    BOOST_TEST(p.release_memory());
    boost::report_errors();
}
