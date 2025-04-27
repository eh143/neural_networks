#include "definitions.h"

// leaky ReLU -> changed to sigmoid because LReLU didn't work
void LReLU(float* dst, float* src, uint32_t size){
    for(int64_t i = 0; i < size; i++){
        __mmask16 opmask = UINT16_MAX >> (i + BITS_IN_UINT16_T - size > 0? i + BITS_IN_UINT16_T - size: 0);

        __m512 data = _mm512_maskz_loadu_ps(opmask, src + i);
        __mmask16 toosmall = _mm512_cmp_ps_mask(data, _mm512_setzero_ps(), _MM_CMPINT_LT);

        data = _mm512_mask_mul_ps(data, toosmall, data, _mm512_set1_ps(0.01f));

        _mm512_mask_storeu_ps(dst + i, opmask, data);

        //dst[i] = 1/(1+expf(-src[i]));
    }
}

// derivative of LReLU!!! -> changed to sigmoid because LReLU didn't work
// defined as the piecewise function:
// / 1      if x >= 0 (done because it's cheaper to do this way on specialized hardware)
// \ -0.01  otherwise
void LReLU_der(float* dst, float* src, uint32_t size){
    __m512 posone = _mm512_set1_ps(1.0f);
    __m512 point01 = _mm512_set1_ps(-0.01f);

    for(int64_t i = 0; i < size; i++){
        __mmask16 opmask = UINT16_MAX >> (i + BITS_IN_UINT16_T - size > 0? i + BITS_IN_UINT16_T - size: 0);

        __m512 data = _mm512_maskz_loadu_ps(opmask, src + i);
        __mmask16 toosmall = _mm512_cmp_ps_mask(data, _mm512_setzero_ps(), _MM_CMPINT_LT);

        __m512 der = _mm512_mask_mov_ps(posone, toosmall, point01);

        _mm512_mask_storeu_ps(dst + i, opmask, der);
        //dst[i] = (1/(1+expf(-src[i])))*(1-(1/(1+expf(-src[i]))));
    }
}

// binary matrix multiplication on an MxN matrix, where M = columns = number of prev neurons and N = rows = number of current neurons
// multiplies weight matrix (weights of current layer) by X vector (activation of previous layer)
void matrix_mul(float* Z, float* X, float* weights, uint32_t M, uint32_t N){
    for(int i = 0; i < N; i++){
        __m512 accumulator = _mm512_setzero_ps();
        for(int64_t j = 0; j < M; j+=BITS_IN_UINT16_T){
            __mmask16 opmask = UINT16_MAX >> (j+BITS_IN_UINT16_T - M > 0? j+BITS_IN_UINT16_T - M: 0);

            __m512 weight = _mm512_maskz_loadu_ps(opmask, weights + i*M + j);
            __m512 activations = _mm512_maskz_loadu_ps(opmask, X + j);

            accumulator = _mm512_add_ps(accumulator, _mm512_mul_ps(weight, activations));
        }

        Z[i] = _mm512_reduce_add_ps(accumulator);
    }
}

void vector_add(float* s0, float* s1, uint32_t size){
    for(int64_t i = 0; i < size; i+=BITS_IN_UINT16_T){
        __mmask16 opmask = UINT16_MAX >> (i + BITS_IN_UINT16_T - size > 0? i + BITS_IN_UINT16_T - size: 0);
        
        __m512 data = _mm512_maskz_loadu_ps(opmask, s0 + i);
        data = _mm512_add_ps(data, _mm512_maskz_loadu_ps(opmask, s1 + i));
        _mm512_mask_storeu_ps(s0 + i, opmask, data);
    }
}

// computes the forward pass for a multi-layered perceptron network NW, with input in of size in_cnt (number of bits)
// NW = network to process
// in = input to network
// in_cnt = size of said input
// output is at NW->L[NW->layer_cnt-1].out
void forward_pass(struct NETWORK* NW, float* in, uint32_t in_cnt){
    matrix_mul(NW->L[0].Z, in, NW->L[0].weights, in_cnt, NW->L[0].neuron_cnt);
    vector_add(NW->L[0].Z, NW->L[0].biases, NW->L[0].neuron_cnt);

    LReLU(NW->L[0].alpha, NW->L[0].Z, NW->L[0].neuron_cnt);

    for(int i = 1; i < NW->layer_cnt; i++){
        matrix_mul(NW->L[i].Z, NW->L[i-1].alpha, NW->L[i].weights, NW->L[i-1].neuron_cnt, NW->L[i].neuron_cnt);
        vector_add(NW->L[i].Z, NW->L[i].biases, NW->L[i].neuron_cnt);

        LReLU(NW->L[i].alpha, NW->L[i].Z, NW->L[i].neuron_cnt);
    }
}

float RandomFloat(float min, float max){
   return ((max - min) * ((float)rand() / RAND_MAX)) + min;
}

// sets up a neural network with lcnt layers (input layer not accounted for) and ncnt neurons per layer
struct NETWORK* init_network(uint32_t* ncnt, uint32_t lcnt, uint32_t input_cnt){
    struct NETWORK* N = calloc(sizeof(struct NETWORK), 1);

    N->layer_cnt = lcnt;
    N->L = calloc(sizeof(struct LAYER), lcnt);

    N->L[0].neuron_cnt = ncnt[0];
    
    N->L[0].alpha = calloc(ncnt[0], sizeof(float));
    N->L[0].biases = calloc(ncnt[0], sizeof(float));
    for(int j = 0; j < ncnt[0]; j++) N->L[0].biases[j] = RandomFloat(0.5f, 2.0f);
    
    N->L[0].Z = calloc(ncnt[0], sizeof(float));

    N->L[0].weights = calloc(ncnt[0]*input_cnt, sizeof(float));
    for(int i = 0; i < ncnt[0]*input_cnt; i++) N->L[0].weights[i] = RandomFloat(-1.0f, 1.0f);

    for(int i = 1; i < lcnt; i++){
        N->L[i].neuron_cnt = ncnt[i];
        
        N->L[i].alpha = calloc(ncnt[i], sizeof(float));
        N->L[i].biases = calloc(ncnt[i], sizeof(float));
        for(int j = 0; j < ncnt[i]; j++) N->L[i].biases[j] = RandomFloat(0.5f, 2.0f);

        N->L[i].Z = calloc(ncnt[i], sizeof(float));

        N->L[i].weights = calloc(ncnt[i]*ncnt[i-1], sizeof(float));
        for(int j = 0; j < ncnt[i]*ncnt[i-1]; j++) N->L[i].weights[j] = RandomFloat(-1.0f, 1.0f);
    }

    return N;
}

void kill_network(struct NETWORK* N){
    for(int i = 0; i < N->layer_cnt; i++){
        free(N->L[i].alpha);
        free(N->L[i].biases);
        free(N->L[i].Z);
        free(N->L[i].weights);
    }
    free(N->L);
    free(N);
}

struct BACKPROP_CTX* init_backprop(uint32_t* ncnt, uint32_t lcnt, uint32_t input_cnt){
    struct BACKPROP_CTX* B = calloc(sizeof(struct NETWORK), 1);

    B->sum = malloc(lcnt*sizeof(float*));
    B->sum[0] = calloc(ncnt[0]*input_cnt, sizeof(float));

    B->error = malloc(lcnt*sizeof(float*));
    B->error[0] = malloc(ncnt[0]*sizeof(float));

    for(int i = 1; i < lcnt; i++){ 
        B->sum[i] = calloc(ncnt[i]*ncnt[i-1], sizeof(float));
        B->error[i] = malloc(ncnt[i]*sizeof(float));
    }
    
    return B;
}

void kill_backprop(struct BACKPROP_CTX* B, uint32_t lcnt){
    for(int i = 0; i < lcnt; i++) free(B->sum[i]);
    for(int i = 0; i < lcnt; i++) free(B->error[i]);
    free(B->sum);
    free(B->error);
    free(B);
}

// transposes matrix by moving all src columns to dst rows 
// src = ptr to source matrix to transpose
// row = number of rows in src matrix
// column = number of columns in src matrix
void* transpose_matrix(float* src, uint64_t row, uint64_t column){
    float* dst = calloc(column*row, sizeof(float));

    for(int i = 0; i < row; i++){
        for(int j = 0; j < column; j++) dst[j*row + i] = src[i*column + j];            
    } 

    return dst;
}
