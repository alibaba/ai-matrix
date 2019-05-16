#include "trtUtil.h"

#include <cassert>
#include <functional>
#include <numeric>

namespace nmtSample
{
int inferTypeToBytes(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kFLOAT: return sizeof(float); break;
    case nvinfer1::DataType::kHALF: return sizeof(int16_t); break;
    default: assert(0); break;
    }
};

int getVolume(nvinfer1::Dims dims)
{
    return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int>());
}

std::vector<float> resizeWeights(int rows, int cols, int rowsNew, int colsNew, const float* memory)
{
    std::vector<float> result(rowsNew * colsNew);
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            result[row * colsNew + col] = memory[row * cols + col];
        }
    }
    return result;
}

} // namespace nmtSample
