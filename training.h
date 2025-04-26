#include "neural_network.h"

// BACKPROPAGATION is regarded by many to be better than evolutive algorithms, except in very specific situations with many local minima

// deltaL = where to store output error
void compute_output_error(float* deltaL_store, float* deltaL_sum, struct NETWORK* N, float* expected_out, __mmask16 clear_old){
    float der[N->L[N->layer_cnt-1].neuron_cnt];
    LReLU_der(der, N->L[N->layer_cnt-1].Z, N->L[N->layer_cnt-1].neuron_cnt);

    for(int64_t i = 0; i < N->L[N->layer_cnt-1].neuron_cnt; i+=BITS_IN_UINT16_T){
        __mmask16 opmask = UINT16_MAX >> (i+BITS_IN_UINT16_T - N->L[N->layer_cnt-1].neuron_cnt > 0? i+BITS_IN_UINT16_T - N->L[N->layer_cnt-1].neuron_cnt: 0);
        __m512 aL = _mm512_maskz_loadu_ps(opmask, N->L[N->layer_cnt-1].alpha + i);
        aL = _mm512_sub_ps(aL, _mm512_maskz_loadu_ps(opmask, expected_out + i));

        __m512 result = _mm512_mul_ps(aL, _mm512_maskz_loadu_ps(opmask, der + i));
        _mm512_mask_storeu_ps(deltaL_store + i, opmask, result);

        __m512 dL_sum = _mm512_maskz_loadu_ps(opmask & (-clear_old), deltaL_sum + i);
        _mm512_mask_storeu_ps(deltaL_sum + i, opmask, _mm512_add_ps(dL_sum, result));
    }
}

// deltaLp1 = delta (error) L plus 1 (of the next layer)
// curLayer = current layer the error is being computed for
void backpropagate_error(float* deltaL_store, float* deltaL_sum, struct NETWORK* N, float* deltaLp1, uint32_t curLayer, __mmask16 clear_old){
    float der[N->L[curLayer].neuron_cnt];
    LReLU_der(der, N->L[curLayer].Z, N->L[curLayer].neuron_cnt);

    float* transposed_weight = transpose_matrix(N->L[curLayer + 1].weights, N->L[curLayer + 1].neuron_cnt, N->L[curLayer].neuron_cnt);

    matrix_mul(deltaL_store, deltaLp1, transposed_weight, N->L[curLayer + 1].neuron_cnt, N->L[curLayer].neuron_cnt);
    free(transposed_weight);

    __mmask16 clearold = -clear_old;
    for(int i = 0; i < N->L[curLayer].neuron_cnt; i+=BITS_IN_UINT16_T){
        __mmask16 overflow = UINT16_MAX >> (i+BITS_IN_UINT16_T - N->L[curLayer].neuron_cnt > 0? i+BITS_IN_UINT16_T - N->L[curLayer].neuron_cnt: 0);

        __m512 dst = _mm512_maskz_loadu_ps(overflow & clearold, deltaL_sum + i);
        __m512 intermediate = _mm512_mul_ps(_mm512_maskz_loadu_ps(overflow, deltaL_store + i), _mm512_maskz_loadu_ps(overflow, der + i));
        dst = _mm512_sub_ps(dst, intermediate);

        _mm512_mask_storeu_ps(deltaL_sum + i, overflow, dst);
    }
}

// in = activations of the first neuron layer (the input)
// in_cnt = number of neurons in first layer
// clear_old = whether to clear before storing (0) or to sum (1), so the previous batches do not affect this one (would be an error) 
void sum_W(struct BACKPROP_CTX* B, struct NETWORK* N, float* in, int64_t in_cnt, __mmask16 clear_old){
    for(int i = 0; i < N->L[0].neuron_cnt; i++){
        for(int j = 0; j < in_cnt; j+=BITS_IN_UINT16_T){
            __m512 error = _mm512_set1_ps(B->error[0][i]);

            __mmask16 opmask = UINT16_MAX >> (j+BITS_IN_UINT16_T - in_cnt > 0? j+BITS_IN_UINT16_T - in_cnt: 0);
            __m512 prev_activations = _mm512_maskz_loadu_ps(opmask, in + j);

            error = _mm512_mul_ps(error, prev_activations);

            __m512 deltaW_sum = _mm512_maskz_loadu_ps(opmask & (-clear_old), B->sum[0] + i*in_cnt + j);

            deltaW_sum = _mm512_add_ps(error, deltaW_sum);
            _mm512_mask_storeu_ps(B->sum[0] + i*in_cnt + j, opmask, deltaW_sum);
        }
    }

    for(int i = 1; i < N->layer_cnt; i++){
        for(int j = 0; j < N->L[i].neuron_cnt; j++){
            for(int k = 0; k < N->L[i-1].neuron_cnt; k+=BITS_IN_UINT16_T){
                __m512 error = _mm512_set1_ps(B->error[i][j]);

                __mmask16 opmask = UINT16_MAX >> (k+BITS_IN_UINT16_T - N->L[i-1].neuron_cnt > 0? k+BITS_IN_UINT16_T - N->L[i-1].neuron_cnt: 0);
                __m512 prev_activations = _mm512_maskz_loadu_ps(opmask, N->L[i-1].alpha + k);

                error = _mm512_mul_ps(error, prev_activations);

                __m512 deltaW_sum = _mm512_maskz_loadu_ps(opmask & (-clear_old), B->sum[i] + j*N->L[i-1].neuron_cnt + k);

                deltaW_sum = _mm512_add_ps(error, deltaW_sum);
                _mm512_mask_storeu_ps(B->sum[i] + j*N->L[i-1].neuron_cnt + k, opmask, deltaW_sum);
            }
        }
    }
}

// M = number of training examples per batch
void update_weights(struct NETWORK* N, struct BACKPROP_CTX* B, struct BACKPROP_CTX* W_SUM, int64_t in_cnt, float eta, uint32_t M){
    __m512 tomul = _mm512_set1_ps(eta/(float)M);

    for(int i = 0; i < N->L[0].neuron_cnt; i++){
        for(int64_t j = 0; j < in_cnt; j+=BITS_IN_UINT16_T){
            __mmask16 opmask = UINT16_MAX >> (j+BITS_IN_UINT16_T - in_cnt > 0? j+BITS_IN_UINT16_T - in_cnt: 0);
            __m512 deltaW = _mm512_maskz_loadu_ps(opmask, W_SUM->sum[0] + i*in_cnt + j);

            deltaW = _mm512_mul_ps(deltaW, tomul);

            __m512 new_W = _mm512_sub_ps(_mm512_maskz_loadu_ps(opmask, N->L[0].weights + i*in_cnt + j), deltaW);
            
            _mm512_mask_storeu_ps(N->L[0].weights + i*in_cnt + j, opmask, new_W); 
        }
    }

    for(int i = 1; i < N->layer_cnt; i++){
        for(int j = 0; j < N->L[i].neuron_cnt; j++){                                                // current number of neurons (rows)
            for(int64_t k = 0; k < N->L[i-1].neuron_cnt; k+=BITS_IN_UINT16_T){                      // previous number of neurons (columns)
                __mmask16 opmask = UINT16_MAX >> (k+BITS_IN_UINT16_T - N->L[i-1].neuron_cnt > 0? j+BITS_IN_UINT16_T - N->L[i-1].neuron_cnt: 0);
                __m512 deltaW = _mm512_maskz_loadu_ps(opmask, W_SUM->sum[i] + j*N->L[i-1].neuron_cnt + k);

                deltaW = _mm512_mul_ps(deltaW, tomul);
                
                __m512 new_W = _mm512_sub_ps(_mm512_maskz_loadu_ps(opmask, N->L[i].weights + j*N->L[i-1].neuron_cnt + k), deltaW);

                _mm512_mask_storeu_ps(N->L[i].weights + j*N->L[i-1].neuron_cnt + k, opmask, new_W); 
            }
        }
    }
}

void update_biases(struct NETWORK* N, struct BACKPROP_CTX* B, float eta, uint32_t M){
    __m512 tomul = _mm512_set1_ps(eta/(float)M);

    for(int i = 0; i < N->layer_cnt; i++){
        for(int64_t j = 0; j < N->L[i].neuron_cnt; j+=BITS_IN_UINT16_T){
            __mmask16 opmask = UINT16_MAX >> (j+BITS_IN_UINT16_T - N->L[i].neuron_cnt > 0? j+BITS_IN_UINT16_T - N->L[i].neuron_cnt: 0);
            __m512 deltaB = _mm512_maskz_loadu_ps(opmask, B->error[i] + j);

            deltaB = _mm512_mul_ps(deltaB, tomul);
            __m512 newB = _mm512_sub_ps(_mm512_maskz_loadu_ps(opmask, N->L[i].biases + j), deltaB);
            _mm512_mask_storeu_ps(N->L[i].biases + j, opmask, newB);
        }
    }
}

// eta_nom = nominator of learning rate
// eta_den = denominator of learning rate
// learning rate is eta_nom/eta_den
// due to PRECISION ISSUES with the vectorized division implementation, we do not recommend having eta_nom be larger than 2**21, or eta_den*batch_size larger than 2**53.
// fixing this "issue" would negatively impact training times.
struct NETWORK* train(uint32_t generations, uint32_t* ncnt, uint32_t lcnt, float** in, uint32_t input_cnt, uint32_t in_instance_size, float** expected_out, uint32_t batch_size, float eta){
    struct NETWORK* N = init_network(ncnt, lcnt, in_instance_size);
    struct BACKPROP_CTX* W_SUM = init_backprop(ncnt, lcnt, in_instance_size);                       // holds sum of all delta(x,L)*(alpha(x,L-1))T within the same batch
    struct BACKPROP_CTX* B = init_backprop(ncnt, lcnt, in_instance_size);                           // holds current EFFECTIVE WEIGHTS and sum of all errors

    for(uint32_t i = 0; i < generations; i++){
// TODO: speed up batch generation somehow
        uint32_t batch[batch_size];
        getrandom(&batch, batch_size*sizeof(uint32_t), 0);
        for(int j = 0; j < batch_size; j++) batch[j] %= input_cnt;

        for(uint32_t j = 0; j < batch_size; j++){
            forward_pass(N, in[batch[j]], in_instance_size);

            compute_output_error(W_SUM->error[N->layer_cnt-1], B->error[N->layer_cnt-1], N, expected_out[batch[j]], j != 0);

// backpropagates the error for all layers
            for(int k = N->layer_cnt - 2; k >= 0 ; k--) backpropagate_error(W_SUM->error[k], B->error[k], N, W_SUM->error[k+1], k, j != 0);
// computes deltaW (how much to change the weights by) and sums it to the backprop context.
            sum_W(W_SUM, N, in[batch[j]], in_instance_size, j != 0);
        }

        update_weights(N, B, W_SUM, in_instance_size, eta, batch_size);
        update_biases(N, B, eta, batch_size);
    }

    kill_backprop(W_SUM, lcnt); 
    kill_backprop(B, lcnt);
    return N;
}
