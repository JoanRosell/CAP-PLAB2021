/*
 * nn.c
 *
 *  Created on: 5 jul. 2016
 *  Author: ecesar
 *
 *      Descripció:
 *      Xarxa neuronal simple de tres capes. La d'entrada que són els pixels d'una
 *      imatge (mirar descripció del format al comentari de readImg) de 32x32 (un total de 1024
 *      entrades). La capa oculta amb un nombre variable de neurones (amb l'exemple proporcionat 117
 *      funciona relativament bé, però si incrementem el nombre de patrons d'entrament caldrà variar-lo).
 *      Finalment, la capa de sortida (que ara té 10 neurones ja que l'entrenem per reconéixer 10
 *      patrons ['0'..'9']).
 *      El programa passa per una fase d'entrenament en la qual processa un conjunt de patrons (en
 *      l'exemple proporcionat són 1934 amb els dígits '0'..'9', escrits a mà). Un cop ha calculat
 *          els pesos entre la capa d'entrada i l'oculta i entre
 *      aquesta i la de sortida, passa a la fase de reconèixament, on llegeix 946 patrons d'entrada
 *      (es proporcionen exemples per aquests patrons), i intenta reconèixer de quin dígit es tracta.
 *
 *  Darrera modificació: gener 2019. Ara l'aprenentatge fa servir la tècnica dels mini-batches
 */

/*******************************************************************************
*    Aquest programa és una adaptació del fet per  JOHN BULLINARIA
*    ( http://www.cs.bham.ac.uk/~jxb/NN/nn.html):
*
*    nn.c   1.0                                       � JOHN BULLINARIA  2004  *
*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include <string.h>
#include <limits.h>
#include <cuda.h>
#include <assert.h>

// Include as extern to link the C source with the CUDA source
extern "C" {
    #include "common.h"
}

// Training set constant row to be used inside the kernels
__constant__ char d_tset_buffer[1025];

// Constants to calculate the different deltas
__constant__ float d_eta = 0.3f;
__constant__ float d_alpha = 0.5f;
__constant__ float d_smallwt = 0.22f;

// Macro to check CUDA errors from syscalls
#define cudaCheckErrors(ans) { __cudaCheckErrors((ans), #ans, __FILE__, __LINE__); }
inline void __cudaCheckErrors(cudaError_t code, const char* call_str, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        char func_name[strlen(call_str)];
        memcpy(func_name, call_str, strlen(call_str));
        char* paren_ptr = strchr(func_name, '(');
        if (paren_ptr != NULL)
        {
            *paren_ptr = '\0';
        }
        fprintf(stderr,"%s: [%s] %s %s:%d\n", func_name, cudaGetErrorName(code), cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define DEBUG

int total;
int seed = 50;
int rando()
{
    seed = (214013 * seed + 2531011);
    return seed >> 16;
}

float frando()
{
    return rando() / 65536.0f;
}

void freeTSet(int np, char** tset)
{
    for (int i = 0; i < np; i++)
    {
        free(tset[i]);
    }
    free(tset);
}

__global__ 
void k_compute_hidden(float* hidden, size_t numHid, float* const weight_ih, size_t numIn, char* const tset)
{
    __shared__ float s_sum[NUMIN]; 
    size_t i = threadIdx.x;

    s_sum[i] = weight_ih[blockIdx.x * blockDim.x + i] * tset[i];
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (i < s)
        {
            s_sum[i] += s_sum[i + s];
        }

        __syncthreads();
    }

    if (i == 0)
    {
        hidden[blockIdx.x] = 1.0 / (1.0 + exp(-s_sum[0]));
    }
}

__global__ 
void k_compute_output(float* output, float* delta_output, size_t numOut, float* const hidden, size_t numHid, float* const weight_ho, float* const target)
{
    __shared__ float s_sum[128]; 
    size_t i = threadIdx.x;

    if (i < NUMHID)
    {
        s_sum[i] = hidden[i] * weight_ho[blockIdx.x * NUMHID + i];
    }
    else
    {
        s_sum[i] = 0.0f;
    }
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (i < s)
        {
            s_sum[i] += s_sum[i + s];
        }

        __syncthreads();
    }

    if (i == 0)
    {
        output[blockIdx.x] = 1.0 / (1.0 + exp(-s_sum[0]));
        delta_output[blockIdx.x] = (target[blockIdx.x] - output[blockIdx.x]) * output[blockIdx.x] * (1.0 - output[blockIdx.x]);    // Sigmoidal Outputs, SSE
    }
}

__global__
void k_compute_batch_error(float* batch_error, float* output, float* target)
{
    __shared__ float s_sum[16];
    size_t i = threadIdx.x;

    if (i < NUMOUT)
    {
        s_sum[i] = 0.5 * (target[i] - output[i]) * (target[i] - output[i]);
    }
    else
    {
        s_sum[i] = 0.0f;
    }

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
    {
        __syncthreads();
        if (i < s)
        {
            s_sum[i] += s_sum[i + s];
        }
    }

    if (i == 0)
    {
        *batch_error = s_sum[0];
    }
}

__global__
void k_compute_delta(float* delta, size_t delta_size, float* in_a, float* in_b)
{
    size_t i = threadIdx.x;
    if (i < delta_size)
    {
        delta[i] = d_eta * in_a[i] * in_b[i] + d_alpha * delta[i];
    }
}

void trainN(const int epochs, const int numIn, const int numHid, const int numOut)
{
    char** tSet;

    float DeltaWeightIH[numHid][numIn], DeltaWeightHO[numOut][numHid];
    float Error, BError, eta = 0.3, alpha = 0.5, smallwt = 0.22;
    int   ranpat[NUMPAT];
    float Hidden[numHid], Output[numOut], DeltaO[numOut], DeltaH[numHid];
    float SumO, SumH, SumDOW;
    float inv_WeightHO[NUMHID][NUMOUT];
    float test_batch_error = 0.0f;

    if ((tSet = loadPatternSet(NUMPAT, "optdigits.tra", 1)) == NULL)
    {
        printf("Loading Patterns: Error!!\n");
        exit(-1);
    }

    // Flatten the training set
    char* h_training_set = (char*) malloc(NUMPAT * 1025 * sizeof(*h_training_set));
    if (h_training_set == NULL)
    {
        perror("malloc");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < NUMPAT; i++)
        for (size_t j = 0; j < 1025; j++)
            h_training_set[i * 1025 + j] = tSet[i][j];

    // Training set
    char* d_training_set;
    cudaCheckErrors(cudaMalloc((void**) &d_training_set, NUMPAT * 1025 * sizeof(*d_training_set)));
    cudaCheckErrors(cudaMemcpy(d_training_set, h_training_set, NUMPAT * 1025 * sizeof(*d_training_set), cudaMemcpyHostToDevice));


    // Init WeightIH
    for (int i = 0; i < numHid; i++)
    {
        for (int j = 0; j < numIn; j++)
        {
            WeightIH[i][j]      = 2.0 * (frando() + 0.01) * smallwt;
            DeltaWeightIH[i][j] = 0.0;
        }
    }

    // Flatten WeightIH
    float* h_weight_ih = (float*) malloc(numHid * numIn * sizeof(*h_weight_ih));
    if (h_weight_ih == NULL)
    {
        perror("malloc");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numHid; i++)
    {
        for (int j = 0; j < numIn; j++)
        {
            h_weight_ih[i * numIn + j] = WeightIH[i][j];
        }
    }

    // Malloc and copy the flattened WeightIH to the device
    float* d_weight_ih;
    cudaCheckErrors(cudaMalloc((void**) &d_weight_ih, numHid * numIn * sizeof(*d_weight_ih)));
    cudaCheckErrors(cudaMemcpy(d_weight_ih, h_weight_ih, numHid * numIn * sizeof(*d_weight_ih), cudaMemcpyHostToDevice));

    // Flatten DeltaWeightIH
    float* h_delta_weight_ih = (float*) malloc(numHid * numIn * sizeof(*h_delta_weight_ih));
    if (h_delta_weight_ih == NULL)
    {
        perror("malloc");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numHid; i++)
    {
        for (int j = 0; j < numIn; j++)
        {
            h_delta_weight_ih[i * numIn + j] = DeltaWeightIH[i][j];
        }
    }

    // Malloc and copy the flattened WeightIH to the device
    float* d_delta_weight_ih;
    cudaCheckErrors(cudaMalloc((void**) &d_delta_weight_ih, numHid * numIn * sizeof(*d_delta_weight_ih)));
    cudaCheckErrors(cudaMemcpy(d_delta_weight_ih, h_delta_weight_ih, numHid * numIn * sizeof(*d_delta_weight_ih), cudaMemcpyHostToDevice));

    // Init WeightHO
    for (int i = 0; i < numOut; i++)
    {
        for (int j = 0; j < numHid; j++)
        {
            WeightHO[i][j]      = 2.0 * (frando() + 0.01) * smallwt;
            DeltaWeightHO[i][j] = 0.0;
            inv_WeightHO[j][i]  = WeightHO[i][j];
        }
    }

    // Flatten WeightHO
    float* h_weight_ho = (float*) malloc(numOut * numHid * sizeof(*h_weight_ho));
    if (h_weight_ho == NULL)
    {
        perror("malloc");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numOut; i++)
    {
        for (int j = 0; j < numHid; j++)
        {
            h_weight_ho[i * numHid + j] = WeightHO[i][j];
        }
    }

    // Malloc and copy the flattened WeightHO to the device
    float* d_weight_ho;
    cudaCheckErrors(cudaMalloc((void**) &d_weight_ho, numOut * numHid * sizeof(*d_weight_ho)));
    cudaCheckErrors(cudaMemcpy(d_weight_ho, h_weight_ho, numOut * numHid * sizeof(*d_weight_ho), cudaMemcpyHostToDevice));

    // Malloc Hidden to the device
    float* d_hidden;
    cudaCheckErrors(cudaMalloc((void**) &d_hidden, numHid * sizeof(*d_hidden)));

    // Malloc Output to the device
    float* d_output;
    cudaCheckErrors(cudaMalloc((void**) &d_output, numOut * sizeof(*d_output)));

    // Malloc Delta Output to the device
    float* d_delta_output;
    cudaCheckErrors(cudaMalloc((void**) &d_delta_output, numOut * sizeof(*d_delta_output)));

    // Flatten Target
    float* h_target = (float*) malloc(NUMPAT * NUMOUT * sizeof(*h_target));
    for (size_t i = 0; i < NUMPAT; i++)
        for (size_t j = 0; j < NUMOUT; j++)
            h_target[i * NUMOUT + j] = Target[i][j];

    // Malloc and copy the Target
    float* d_target;
    cudaCheckErrors(cudaMalloc((void**) &d_target, NUMPAT * NUMOUT * sizeof(*d_target)));
    cudaCheckErrors(cudaMemcpy(d_target, h_target, NUMPAT * NUMOUT * sizeof(*d_target), cudaMemcpyHostToDevice));

    // Declare batch error in the device
    float* d_batch_error;
    cudaCheckErrors(cudaMalloc((void**) &d_batch_error, sizeof(*d_batch_error)));

    // Malloc DeltaH in the device
    float* d_delta_h; 
    cudaCheckErrors(cudaMalloc((void**) &d_delta_h, numHid * sizeof(*d_delta_h)));

    Error = 10;
    for (int epoch = 0; epoch < epochs && Error >= 0.0004; epoch++) // iterate weight updates
    {
        for (int p = 0; p < NUMPAT; p++) // randomize order of individuals
        {
            ranpat[p] = p;
        }
        for (int p = 0; p < NUMPAT; p++)
        {
            int x  = rando();
            int np = (x * x) % NUMPAT;
            int op = ranpat[p];
            ranpat[p]  = ranpat[np];
            ranpat[np] = op;
        }

        printf(".");
        fflush(stdout);

        float tmp_batch_error;
        Error = 0.0;
        for (int nb = 0; nb < NUMPAT / BSIZE; nb++) // repeat for all batches
        {
            BError = 0.0;
            test_batch_error = 0.0f;
            for (int np = nb * BSIZE; np < (nb + 1) * BSIZE; np++) // repeat for all the training patterns within the batch
            {
                int p = ranpat[np];

                // This copy from global to const memory should be amortized by the next kernels, for now we will use global mem
                //cudaCheckErrors(cudaMemcpyToSymbol(d_tset_buffer, &d_training_set[p * 1025], 1025 * sizeof(*d_training_set), 0, cudaMemcpyDeviceToDevice));

                // Kernel 1: Sequential
                for (int j = 0; j < numHid; j++) // compute hidden unit activations
                {
                    float SumH = 0.0;
                    for (int i = 0; i < numIn; i++)
                    {
                        SumH += h_weight_ih[j * numIn + i] * h_training_set[p * 1025 + i];
                    }
                    Hidden[j] = 1.0 / (1.0 + exp(-SumH));
                }
                k_compute_hidden<<<numHid, numIn>>>(d_hidden, NUMHID, d_weight_ih, NUMIN, &d_training_set[p * 1025]);
                cudaCheckErrors(cudaGetLastError());
                cudaCheckErrors(cudaMemcpy(Hidden, d_hidden, sizeof(*Hidden) * NUMHID, cudaMemcpyDeviceToHost));

                #ifdef DEBUG
                float test_hidden[NUMHID] = {0};
                for (int j = 0; j < numHid; j++) // compute hidden unit activations
                {
                    float SumH = 0.0f;
                    for (int i = 0; i < numIn; i++)
                    {
                        SumH += h_weight_ih[j * numIn + i] * h_training_set[p * 1025 + i];
                    }
                    test_hidden[j] = 1.0f / (1.0f + exp(-SumH));
                }

                for (size_t h = 0; h < numHid; h++)
                {
                    if (abs(Hidden[h] - test_hidden[h]) > 0.0001f)
                    {
                        printf("GPU error while computing HIDDEN @ idx: %lu\n", h);
                        printf("\tCPU val: %f\n\tGPU val: %f\n", test_hidden[h], Hidden[h]);
                        exit(EXIT_FAILURE);
                    }
                }
                #endif

/*
for (int k = 0; k < numOut; k++) // compute output unit activations and errors
{
float SumO = 0.0;
for (int j = 0; j < numHid; j++)
{
SumO += Hidden[j] * d_weight_ho[k][j];
}
Output[k] = 1.0 / (1.0 + exp(-SumO));                                      // Sigmoidal Outputs
BError   += 0.5 * (Target[p][k] - Output[k]) * (Target[p][k] - Output[k]); // SSE
DeltaO[k] = (Target[p][k] - Output[k]) * Output[k] * (1.0 - Output[k]);    // Sigmoidal Outputs, SSE
}
                */

                k_compute_output<<<numOut, 128>>>(d_output, d_delta_output, numOut, d_hidden, numHid, d_weight_ho, &d_target[p * NUMOUT]); 
                cudaCheckErrors(cudaGetLastError());

                k_compute_batch_error<<<1, 16>>>(d_batch_error, d_output, &d_target[p * NUMOUT]);
                cudaCheckErrors(cudaGetLastError());
                cudaCheckErrors(cudaMemcpy(Output, d_output, sizeof(*Output) * numOut, cudaMemcpyDeviceToHost));
                cudaCheckErrors(cudaMemcpy(DeltaO, d_delta_output, sizeof(*DeltaO) * numOut, cudaMemcpyDeviceToHost));
                cudaCheckErrors(cudaMemcpy(&tmp_batch_error, d_batch_error, sizeof(BError), cudaMemcpyDeviceToHost));
                BError += tmp_batch_error;

                #ifdef DEBUG
                float test_output[NUMOUT];
                float test_delta_output[NUMOUT];
                for (int k = 0; k < numOut; k++) // compute output unit activations and errors
                {
                    float SumO = 0.0;
                    for (int j = 0; j < numHid; j++)
                    {
                        SumO += Hidden[j] * h_weight_ho[k * numHid + j];
                    }
                    test_output[k] = 1.0 / (1.0 + exp(-SumO));                                      // Sigmoidal Outputs
                    test_batch_error   += 0.5 * (h_target[p * NUMOUT + k] - test_output[k]) * (h_target[p * NUMOUT + k] - test_output[k]); // SSE
                    test_delta_output[k] = (h_target[p * NUMOUT + k] - test_output[k]) * test_output[k] * (1.0 - test_output[k]);    // Sigmoidal Outputs, SSE
                }

                for (size_t i = 0; i < numOut; i++)
                {
                    if (abs(Output[i] - test_output[i]) > 0.0001f)
                    {
                        printf("GPU error while computing Output @ idx: %lu\n", i);
                        printf("\tCPU val: %f\n\tGPU val: %f\n", test_output[i], Output[i]);
                        exit(EXIT_FAILURE);
                    }
                }

                for (size_t i = 0; i < numOut; i++)
                {
                    if (abs(DeltaO[i] - test_delta_output[i]) > 0.0001f)
                    {
                        printf("GPU error while computing DeltaO @ idx: %lu\n", i);
                        printf("\tCPU val: %f\n\tGPU val: %f\n", test_delta_output[i], DeltaO[i]);
                        exit(EXIT_FAILURE);
                    }
                }

                if (abs(BError - test_batch_error) > 0.0001f)
                {
                    printf("GPU error while computing BError\n");
                    printf("\tCPU val: %f\n\tGPU val: %f\n", test_batch_error, BError);
                    exit(EXIT_FAILURE);
                }
                #endif

                for (int j = 0; j < numHid; j++)                                               // update delta weights DeltaWeightIH
                {
                    float SumDOW = 0.0;
                    for (int k = 0; k < numOut; k++)
                    {
                        SumDOW += h_weight_ho[k * numHid + j] * DeltaO[k];
                    }
                    DeltaH[j] = SumDOW * Hidden[j] * (1.0 - Hidden[j]);
                }

                /*
                for (int j = 0; j < numHid; j++)                                               // update delta weights DeltaWeightIH
                {
                    for (int i = 0; i < numIn; i++)
                    {
                        h_delta_weight_ih[j][i] = eta * DeltaH[j] * h_training_set[p * 1025 + i] + alpha * h_delta_weight_ih[j][i];
                    }
                }
                */

                cudaCheckErrors(cudaMemcpy(d_delta_h, DeltaH, numHid * sizeof(*d_delta_h), cudaMemcpyHostToDevice));
                k_compute_delta<<<numHid, numIn>>>(d_delta_weight_ih, numIn, d_delta_h, d_training_set[p * 1025]);
                cudaCheckErrors(cudaGetLastError());
                cudaCheckErrors(cudaMemcpy(h_delta_weight_ih, d_delta_weight_ih, numIn * sizeof(*d_delta_weight_ih), cudaMemcpyDeviceToHost));

                for (int k = 0; k < numOut; k++) // update delta weights DeltaWeightHO
                {
                    for (int j = 0; j < numHid; j++)
                    {
                        DeltaWeightHO[k][j] = eta * Hidden[j] * DeltaO[k] + alpha * DeltaWeightHO[k][j];
                    }
                }
            }

            for (int j = 0; j < numHid; j++) // update weights WeightIH
            {
                for (int i = 0; i < numIn; i++)
                {
                    h_weight_ih[j * numIn + i] += h_delta_weight_ih[j * numIn + i];
                }
            }
            cudaCheckErrors(cudaMemcpy(d_weight_ih, h_weight_ih, numHid * numIn * sizeof(*d_weight_ih), cudaMemcpyHostToDevice));

            for (int k = 0; k < numOut; k++) // update weights WeightHO
            {
                for (int j = 0; j < numHid; j++)
                {
                    h_weight_ho[k * numHid + j]    += DeltaWeightHO[k][j];
                }
            }
            cudaCheckErrors(cudaMemcpy(d_weight_ho, h_weight_ho, numOut * numHid * sizeof(*d_weight_ho), cudaMemcpyHostToDevice));

            Error += BError; // We only want to update Error once per iteration
        }


        Error = Error / ((NUMPAT / BSIZE) * BSIZE); //mean error for the last epoch
        if (!(epoch % 100))
        {
            printf("\nEpoch %-5d :   Error = %f \n", epoch, Error);
        }
        if (Error < 0.0004)
        {
            printf("\nEpoch %-5d :   Error = %f \n", epoch, Error);
        }
    }

    for (size_t i = 0; i < numHid; i++)
        for (size_t j = 0; j < numIn; j++)
            WeightIH[i][j] = h_weight_ih[i * numIn + j];

    for (size_t i = 0; i < numOut; i++)
        for (size_t j = 0; j < numHid; j++)
            WeightHO[i][j] = h_weight_ho[i * numHid + j];

    freeTSet(NUMPAT, tSet);
    free(h_training_set);
    free(h_weight_ih);
    printf("END TRAINING\n");
}

/*
void _trainN(const int epochs, const int numIn, const int numHid, const int numOut)
{
    char** tSet;

    float DeltaWeightIH[numHid][numIn], DeltaWeightHO[numOut][numHid];
    // TODO: load eta, alpha and smallwt to constant device memory
    float Error, BError, eta = 0.3, alpha = 0.5, smallwt = 0.22;
    int   ranpat[NUMPAT];
    float Hidden[numHid], Output[numOut], DeltaO[numOut], DeltaH[numHid];
    float SumO, SumH, SumDOW;
    float inv_WeightHO[NUMHID][NUMOUT]; // TODO: malloc this array so we can use the inverse in the device

    if ((tSet = loadPatternSet(NUMPAT, "optdigits.tra", 1)) == NULL)
    {
        printf("Loading Patterns: Error!!\n");
        exit(-1);
    }

    uint8_t* flat_tset = (uint8_t*) malloc(NUMPAT * 1025 * sizeof(*flat_tset));
    for (size_t i = 0; i < NUMPAT; i++)
    {
        for (size_t j = 0; j < 1025; j++)
            flat_tset[i * 1025 + j] = tSet[i][j];
    }

    for (int i = 0; i < numHid; i++)
    {
        for (int j = 0; j < numIn; j++)
        {
            WeightIH[i][j]      = 2.0 * (frando() + 0.01) * smallwt;
            DeltaWeightIH[i][j] = 0.0;
        }
    }

    for (int i = 0; i < numOut; i++)
    {
        for (int j = 0; j < numHid; j++)
        {
            WeightHO[i][j]      = 2.0 * (frando() + 0.01) * smallwt;
            DeltaWeightHO[i][j] = 0.0;
            inv_WeightHO[j][i]  = WeightHO[i][j];
        }
    }

    // Se reserva el espacio de memoria en la GPU
    // WeightIH
    float* d_WeightIH;
    float* flat_weight_ih = (float*) malloc(numHid * numIn * sizeof(float));

    for (size_t i = 0; i < numHid; i++)
    {
        for (size_t j = 0; j < numIn; j++)
        {
            flat_weight_ih[i * numIn + j] = WeightIH[i][j];
        }
    }

    cudaCheckErrors(cudaMalloc((void**) &d_WeightIH, numHid * numIn * sizeof(float)));
    cudaCheckErrors(cudaMemcpy(d_WeightIH, flat_weight_ih, numHid * numIn * sizeof(float), cudaMemcpyHostToDevice));

    // tSet
    uint8_t* d_flat_tset;

    cudaCheckErrors(cudaMalloc((void**) &d_flat_tset, NUMPAT * 1025 * sizeof(*d_flat_tset)));
    cudaCheckErrors(cudaMemcpy(d_flat_tset, flat_tset, NUMPAT * 1025 * sizeof(*d_flat_tset), cudaMemcpyHostToDevice));

    // Hidden
    float* d_Hidden;

    cudaCheckErrors(cudaMalloc((void**) &d_Hidden, numHid * sizeof(float)));

    // WeightHO
    float* d_WeightHO;
    float* flat_weight_ho = (float*) malloc(numOut * numHid * sizeof(float));

    for (size_t i = 0; i < numOut; i++)
    {
        for (size_t j = 0; j < numHid; j++)
        {
            flat_weight_ho[i * numHid + j] = WeightHO[i][j];
        }
    }

    cudaCheckErrors(cudaMalloc((void**) &d_WeightHO, numOut * numHid * sizeof(float)));
    cudaCheckErrors(cudaMemcpy(d_WeightHO, flat_weight_ho, numOut * numHid * sizeof(float), cudaMemcpyHostToDevice));

    // Output
    float* d_Output;

    cudaCheckErrors(cudaMalloc((void**) &d_Output, numOut * sizeof(float)));

    // Target
    float* d_Target;
    float* flat_target = (float*) malloc(NUMPAT * NUMOUT * sizeof(float));

    for (size_t i = 0; i < NUMPAT; i++)
    {
        for (size_t j = 0; j < NUMOUT; j++)
        {
            flat_target[i * NUMOUT + j] = Target[i][j];
        }
    }

    cudaCheckErrors(cudaMalloc((void**) &d_Target, NUMPAT * NUMOUT * sizeof(float)));
    cudaCheckErrors(cudaMemcpy(d_Target, flat_target, NUMPAT * NUMOUT * sizeof(float), cudaMemcpyHostToDevice));

    //cudaMalloc((void**) &d_tSet_msk, NUMPAT * 1024 * sizeof(uint32_t));
    //cudaMalloc((void**) &d_WeightHO, NUMOUT * NUMHID * sizeof(float));
    //cudaMalloc((void**) &d_Output, numOut * sizeof(float));
    //cudaMalloc((void**) &d_Target, NUMPAT * NUMOUT * sizeof(float));
    //cudaMalloc((void**) &d_DeltaO, numOut * sizeof(float));
    //cudaMalloc((void**) &d_inv_Weight, NUMOUT * NUMHID * sizeof(float));
    //cudaMalloc((void**) &d_DeltaH, numHid * sizeof(float));
    //cudaMalloc((void**) &d_WeightIH, NUMHID * NUMIN * sizeof(float));
    //cudaMalloc((void**) &d_WeigthHO, NUMOUT * NUMHID * sizeof(float));

    Error = 10;
    for (int epoch = 0; epoch < epochs && Error >= 0.0004; epoch++) // iterate weight updates
    {
        for (int p = 0; p < NUMPAT; p++)                            // randomize order of individuals
        {
            ranpat[p] = p;
        }

        for (int p = 0; p < NUMPAT; p++)
        {
            int x  = rando();
            int np = (x * x) % NUMPAT;
            int op = ranpat[p];
            ranpat[p]  = ranpat[np];
            ranpat[np] = op;
        }

        printf(".");
        fflush(stdout);

        Error = 0.0;
        for (int nb = 0; nb < NUMPAT / BSIZE; nb++) // repeat for all batches
        {
            BError = 0.0;
            for (int np = nb * BSIZE; np < (nb + 1) * BSIZE; np++) // repeat for all the training patterns within the batch
            {
                int p = ranpat[np];

                k_compute_hidden<<<numHid, numIn>>>(d_Hidden, NUMHID, d_WeightIH, NUMIN, &d_flat_tset[p * 1025]);
                cudaError_t errSync  = cudaGetLastError();
                if (errSync != cudaSuccess) 
                {
                    printf("\nSync kernel error: %s\n", cudaGetErrorString(errSync));
                    exit(EXIT_FAILURE);
                }

                cudaCheckErrors(cudaMemcpy(Hidden, d_Hidden, sizeof(*Hidden) * NUMHID, cudaMemcpyDeviceToHost));

                #ifdef DEBUG
                float test_hidden[NUMHID] = {0};
                for (int j = 0; j < numHid; j++) // compute hidden unit activations
                {
                    float SumH = 0.0f;
                    for (int i = 0; i < numIn; i++)
                    {
                        SumH += flat_weight_ih[j * numIn + i] * flat_tset[p * 1025 + i];
                    }
                    test_hidden[j] = 1.0f / (1.0f + exp(-SumH));
                }

                for (size_t h = 0; h < numHid; h++)
                {
                    if (abs(Hidden[h] - test_hidden[h]) > 0.0001f)
                    {
                        printf("GPU error while computing HIDDEN @ idx: %lu\n", h);
                        printf("\tCPU val: %f\n\tGPU val: %f\n", test_hidden[h], Hidden[h]);
                        exit(EXIT_FAILURE);
                    }
                }
                #endif

                for (int k = 0; k < numOut; k++) // compute output unit activations and errors
                {
                    float SumO = 0.0;
                    for (int j = 0; j < numHid; j++)
                    {
                        SumO += Hidden[j] * flat_weight_ho[k * numHid + j];
                    }
                    Output[k] = 1.0 / (1.0 + exp(-SumO));                                      // Sigmoidal Outputs
                    BError   += 0.5 * (Target[p][k] - Output[k]) * (Target[p][k] - Output[k]); // SSE
                    DeltaO[k] = (Target[p][k] - Output[k]) * Output[k] * (1.0 - Output[k]);    // Sigmoidal Outputs, SSE
                }

                for (int j = 0; j < numHid; j++)                                               // update delta weights DeltaWeightIH
                {
                    float SumDOW = 0.0;
                    for (int k = 0; k < numOut; k++)
                    {
                        SumDOW += flat_weight_ho[j * numOut + k] * DeltaO[k];
                    }

                    DeltaH[j] = SumDOW * Hidden[j] * (1.0 - Hidden[j]);
                    for (int i = 0; i < numIn; i++)
                    {
                        //DeltaWeightIH[j][i] = f_and(eta * DeltaH[j], tSet_msk[p * 1024 + i]) + alpha * DeltaWeightIH[j][i];
                        DeltaWeightIH[j][i] = (eta * DeltaH[j]) * flat_tset[p * 1025 + i] + alpha * DeltaWeightIH[j][i];
                    }
                }

                for (int k = 0; k < numOut; k++) // update delta weights DeltaWeightHO
                {
                    for (int j = 0; j < numHid; j++)
                    {
                        DeltaWeightHO[k][j] = eta * Hidden[j] * DeltaO[k] + alpha * DeltaWeightHO[k][j];
                    }
                }
            }

            for (int j = 0; j < numHid; j++) // update weights WeightIH
            {
                for (int i = 0; i < numIn; i++)
                {
                    flat_weight_ih[j * numIn + i] += DeltaWeightIH[j][i];
                }
            }
            cudaCheckErrors(cudaMemcpy(d_WeightIH, flat_weight_ih, numHid * numIn * sizeof(float), cudaMemcpyHostToDevice));

            for (int k = 0; k < numOut; k++) // update weights WeightHO
            {
                for (int j = 0; j < numHid; j++)
                {
                    flat_weight_ho[k * numHid + j] += DeltaWeightHO[k][j];
                    inv_WeightHO[j][k]              = flat_weight_ho[k * numHid + j];
                }
            }

            Error += BError; // We only want to update Error once per iteration
        }


        Error = Error / ((NUMPAT / BSIZE) * BSIZE); //mean error for the last epoch
        if (!(epoch % 100))
        {
            printf("\nEpoch %-5d :   Error = %f \n", epoch, Error);
        }
        if (Error < 0.0004)
        {
            printf("\nEpoch %-5d :   Error = %f \n", epoch, Error);
        }
    }

    // Return the train results to the original arrays
    for (size_t i = 0; i < numHid; i++)
    {
        for (size_t j = 0; j < numIn; j++)
        {
            WeightIH[i][j] = flat_weight_ih[i * numIn + j];
        }
    }

    for (size_t i = 0; i < numOut; i++)
    {
        for (size_t j = 0; j < numHid; j++)
        {
            WeightHO[i][j] = flat_weight_ho[i * numHid + j];
        }
    }

    freeTSet(NUMPAT, tSet);
    printf("END TRAINING\n");
}
*/

void printRecognized(int p, float Output[], const int numOut)
{
    int imax = 0;

    for (int i = 1; i < numOut; i++)
    {
        if (Output[i] > Output[imax])
        {
            imax = i;
        }
    }
    printf("El patró %d sembla un %c\t i és un %d", p, '0' + imax, Validation[p]);
    if (imax == Validation[p])
    {
        total++;
    }
    for (int k = 0; k < numOut; k++)
    {
        printf("\t%f\t", Output[k]);
    }
    printf("\n");
}

void runN(const int numIn, const int numHid, const int numOut)
{
    char** rSet;
    char*  fname[NUMRPAT];

    if ((rSet = loadPatternSet(NUMRPAT, "optdigits.cv", 0)) == NULL)
    {
        printf("Error!!\n");
        exit(-1);
    }

    float Hidden[numHid], Output[numOut];

    for (int p = 0; p < NUMRPAT; p++)    // repeat for all the recognition patterns
    {
        for (int j = 0; j < numHid; j++) // compute hidden unit activations
        {
            float SumH = 0.0;
            for (int i = 0; i < numIn; i++)
            {
                SumH += rSet[p][i] * WeightIH[j][i];
            }
            Hidden[j] = 1.0 / (1.0 + exp(-SumH));
        }

        for (int k = 0; k < numOut; k++) // compute output unit activations
        {
            float SumO = 0.0;
            for (int j = 0; j < numHid; j++)
            {
                SumO += Hidden[j] * WeightHO[k][j];
            }
            Output[k] = 1.0 / (1.0 + exp(-SumO)); // Sigmoidal Outputs
        }
        printRecognized(p, Output, numOut);
    }

    printf("\nTotal encerts = %d\n", total);

    freeTSet(NUMRPAT, rSet);
}

int main(int argc, char** argv)
{
    // Read parameters from CLI
    const int epochs = (argc > 1) ? atoi(argv[1]) : 1000000;
    const int numIn  = (argc > 2) ? atoi(argv[2]) : NUMIN;
    const int numHid = (argc > 3) ? atoi(argv[3]) : NUMHID;
    const int numOut = (argc > 4) ? atoi(argv[4]) : NUMOUT;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device props:\n");
    printf("    Mode: %d\n", prop.computeMode);
    printf("    Capability: %d.%d\n", prop.minor, prop.major);
    printf("\n");

    clock_t start = clock();

    trainN(epochs, numIn, numHid, numOut);
    runN(numIn, numHid, numOut);

    clock_t end = clock();

    printf("\n\nGoodbye! (%f sec)\n\n", (end - start) / (1.0 * CLOCKS_PER_SEC));

    exit(EXIT_SUCCESS);
}

/******************************************************************************/
