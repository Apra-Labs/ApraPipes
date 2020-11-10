/* Copyright (C) 2011 John Maddock
* 
* Use, modification and distribution is subject to the 
* Boost Software License, Version 1.0. (See accompanying
* file LICENSE_1_0.txt or http://www.boost.org/LICENSE_1_0.txt)
*/

// 
// Verify that if malloc/free are macros that everything still works OK:
//

#include <functional>
#include <new>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <algorithm>
#include <boost/limits.hpp>
#include <iostream>
#include <locale>

namespace std{

   void* undefined_poisoned_symbol1(unsigned x);
   void undefined_poisoned_symbol2(void* x);

}

#define malloc(x) undefined_poisoned_symbol1(x)
#define free(x) undefined_poisoned_symbol2(x)

#include "ApraPool.h"
#include <boost/pool/pool.hpp>

template class ApraPool<boost::default_user_allocator_new_delete>;
template class ApraPool<boost::default_user_allocator_malloc_free>;

