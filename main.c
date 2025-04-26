#include "training.h"
#include "stdio.h"

#define TRAINING_SET 4

int main(){
    float in[4][2] = {{0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}};
    float out[4][1] = {{1.0f}, {1.0f}, {1.0f}, {0.0f}};

    float** real_in = calloc(4, sizeof(float*)), **real_out = calloc(4, sizeof(float*));

    for(int i = 0; i < 4; i++){
        real_in[i] = calloc(2, sizeof(float));
        real_in[i][0] = in[i][0];
        real_in[i][1] = in[i][1];

        real_out[i] = calloc(1, sizeof(float));
        real_out[i][0] = out[i][0];
    }

    uint32_t ncnt[] = {4, 4, 1};

    int a = 0;
    getrandom(&a, sizeof(int), 0);
    a %= 65536;
    for(int i = 0; i < a; i++) printf("%f ", RandomFloat(-1, 1));
    puts("");

    struct NETWORK* N = train(10, ncnt, 3, real_in, 4, 2, real_out, 4, 0.75);

// cleanup
    kill_network(N);
    for(int i = 0; i < 4; i++){
        free(real_in[i]); 
        free(real_out[i]);
    }
    free(real_in); free(real_out);

    return 0;
}
