#ifndef SAMPLE_NMT_TRT_UTIL_
#define SAMPLE_NMT_TRT_UTIL_

#include "NvInfer.h"
#include <vector>

namespace nmtSample
{
int inferTypeToBytes(nvinfer1::DataType t);

int getVolume(nvinfer1::Dims dims);

// Resize weights matrix to larger size
std::vector<float> resizeWeights(int rows, int cols, int rowsNew, int colsNew, const float* memory);

} // namespace nmtSample

#endif // SAMPLE_NMT_TRT_UTIL_
