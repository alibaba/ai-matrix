#ifndef SAMPLE_NMT_SLP_PROJECTION_
#define SAMPLE_NMT_SLP_PROJECTION_

#include "projection.h"
#include "trtUtil.h"

#include "componentWeights.h"

extern int gPadMultiple;

namespace nmtSample
{
/** \class SLPProjection
    *
    * \brief Linear logits calculation
    *
    * Calculates logits vector by multiplying input vector with weight matrix  
    *
    */
class SLPProjection : public Projection
{
public:
    SLPProjection(ComponentWeights::ptr weights);

    void addToModel(
        nvinfer1::INetworkDefinition* network,
        nvinfer1::ITensor* input,
        nvinfer1::ITensor** outputLogits) override;

    int getOutputSize() override;

    std::string getInfo() override;

    ~SLPProjection() override = default;

protected:
    ComponentWeights::ptr mWeights;
    nvinfer1::Weights mKernelWeights;
    int mInputChannelCount;
    int mOutputChannelCount;
    std::vector<float> mResizedKernelWeights;
};
} // namespace nmtSample

#endif // SAMPLE_NMT_SLP_PROJECTION_
