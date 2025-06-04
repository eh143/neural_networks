#include "training.h"
#include "stdio.h"

#define TRAINING_SET 4

int main(){
    uint32_t ncnt[] = {200, 100, 80, 10};

    int a = 0;
    getrandom(&a, sizeof(int), 0);
    srand(a);
    load_mnist();


    struct NETWORK* N = init_network(ncnt, 4, SIZE);

    FILE* fd = fopen("highaccuracy.bin", "r+b");
        
    fread(N->L[0].weights, sizeof(float), SIZE*N->L[0].neuron_cnt, fd);
    fread(N->L[0].biases, sizeof(float), N->L[0].neuron_cnt, fd);
    
    for(int j = 1; j < N->layer_cnt; j++){
        fread(N->L[j].weights, sizeof(float), N->L[j-1].neuron_cnt*N->L[j].neuron_cnt, fd);
        fread(N->L[j].biases, sizeof(float), N->L[j].neuron_cnt, fd);
    }
    fclose(fd);

    int correct_guesses = 0;
    for(int i = 0; i < NUM_TEST; i++){
        forward_pass(N, test_image[i], SIZE);

        float highest_hyp = 0, highest_test = 0;
        int ind_hyp = 0, ind_test = 0;

        for(int k = 0; k < 10; k++){
            if(highest_test < test_label[i][k]){ highest_test = test_label[i][k]; ind_test = k; }
            if(highest_hyp < N->L[N->layer_cnt-1].alpha[k]){ highest_hyp = N->L[N->layer_cnt-1].alpha[k]; ind_hyp = k; }
        }

        if(ind_test == ind_hyp) correct_guesses++;
        else{ printf("%d %d\n", ind_test, ind_hyp); print_number(test_image[i]); }
    }

    printf("ACCURACY IS %f\n", ((float)correct_guesses/NUM_TEST)*100);


    kill_network(N);

    return 0;
}
