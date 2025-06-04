#include "stdint.h"
#include "string.h"
#include "immintrin.h"
#include "sys/random.h"
#include "math.h"
#include "stdio.h"
#include "MNIST_for_C-master/mnist.h"

#define BITS_IN_UINT64_T 64
#define BITS_IN_UINT16_T 16

struct LAYER{
    float* weights;                     // weight matrix
    float* biases;                      // bias vector
    float* Z;                           // output pre-activation function
    float* alpha;                       // output post-activation function
    uint32_t neuron_cnt;
};

struct NETWORK{
    struct LAYER* L;
    uint32_t layer_cnt;
};

struct BACKPROP_CTX{
    float** sum;                        // sum of values that later becomes the weight matrix
    float** error;                      // where errors are stored n shit
};
