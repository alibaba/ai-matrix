# AI Matrix

AI Matrix aims at measuring the performance of AI hardware platforms and software frameworks. The benchmark suite currently consists of three types of workloads: layer-based benchmark, macro benchmark, and micro benchmark.

## Layer-Based Benchmark
Layer-based benchmark consists of commonly used layers in deep learning neural networks, such as convolution layer, fully-connected layer, activation layer, etc. The purpose of layer-based benchmark is to test the performance of AI hardware at running the commonly used neural network layers.

## Macro Benchmark
Macro benchmark consists of a number of full models widely used in deep learning neural networks. These models cover the most common AI application fields. This benchmark aims at measuring the performance of hardware platforms at training and infering real deep learning models. Macro benchmark also includes some innovative AI algorithms widely deployed inside Alibaba. More information about the features of this benchmark suite can be found [**here**](http://aimatrix.ai/#!/docs/goals.md?lang=en-us).

## Micro Benchmark
Micro benchmark mainly consists of workloads of matrix operations. The benchmark comes from Baidu's DeepBench. The purpose of the benchmark is to test the hardware platform's ability to do matrix computations.

## Run the Benchmarks
To run the benchmark in each of the three categories, please follow the instructions in the README.md file in each benchmark directory.

## Suggestions and Questions
AI Matrix is still in early development stage and the development team is working hard to make it better and better. Any suggestions on improving the benchmark suite are highly welcomed and appreciated. If you have questions, good suggestions, or want to participate, please do not hesitate to contact us. You could submit questions on Github or contact us through aimatrix@list.alibaba-inc.com

## Attention
As this repository has some large files, please install [git lfs](https://github.com/git-lfs/git-lfs/wiki/Installation) before you start to download it.
