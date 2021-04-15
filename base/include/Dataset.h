#pragma once

#include <vector>

class Dataset
{
public:
    std::vector<void *> data;
    std::vector<size_t> size;
};