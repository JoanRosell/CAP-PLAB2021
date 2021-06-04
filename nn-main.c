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
#include <mpi/mpi.h>
//#include <mpi.h>
#include <omp.h>

#include "common.h"

// Constants
static const float k_eta = 0.3;
static const float k_alpha = 0.5;
static const float k_smallwt = 0.22;

// Globals
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

float f_and(float val, uint32_t msk)
{
    uint32_t tmp = 0;

    memcpy(&tmp, &val, 4);
    tmp &= msk;
    memcpy(&val, &tmp, 4);

    return val;
}

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
    //printf("El patró %d sembla un %c\t i és un %d", p, '0' + imax, Validation[p]);
    if (imax == Validation[p])
    {
        total++;
    }
    for (int k = 0; k < numOut; k++)
    {
        //printf("\t%f\t", Output[k]);
    }
    //printf("\n");
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

struct batch_chunk
{
    size_t start;
    size_t end;
};

void randomize_pattern_order(int* const patterns)
{
    for (int p = 0; p < NUMPAT; p++)
    {
        patterns[p] = p;
    }

    for (int p = 0; p < NUMPAT; p++)
    {
        int x  = rando();
        int np = (x * x) % NUMPAT;
        int op = patterns[p];
        patterns[p]  = patterns[np];
        patterns[np] = op;
    }
}

void init_weight(float* const weight, size_t num_rows, size_t num_cols)
{
    for (int i = 0; i < num_rows; i++)
    {
        for (int j = 0; j < num_cols; j++)
        {
            weight[i * num_cols + j] = 2.0 * (frando() + 0.01) * k_smallwt;
        }
    }
}

void init_delta_weight(float* const delta_weight, size_t size)
{
    memset(delta_weight, 0, size * sizeof(*delta_weight));
}

void trainN(int my_rank, int nprocs, const int epochs, int argc, char** argv, const int numIn, const int numHid, const int numOut)
{
    int   ranpat[NUMPAT];
    float Hidden[numHid], Output[numOut], DeltaO[numOut], DeltaH[numHid];

    size_t input_hidden_size = NUMHID * NUMIN;
    size_t hidden_output_size = NUMOUT * NUMHID;

    char** tSet = NULL;
    if ((tSet = loadPatternSet(NUMPAT, "optdigits.tra", 1)) == NULL)
    {
        printf("Loading Patterns: Error!!\n");
        exit(-1);
    }

    uint32_t* tSet_msk = malloc(sizeof(uint32_t) * NUMPAT * 1024);
    for (size_t i = 0; i < NUMPAT; i++)
    {
        for (size_t j = 0; j < 1024; j++)
        {
            tSet_msk[i * 1024 + j] = tSet[i][j] * 0xFFFFFFFF;
        }
    }

    // Initialize each weights array
    float* weight_ih = malloc(input_hidden_size * sizeof(*weight_ih));
    if (weight_ih == NULL)
    {
        perror("malloc");
        exit(EXIT_FAILURE);
    }
    init_weight(weight_ih, NUMHID, NUMIN);

    float* delta_weight_ih = malloc(input_hidden_size * sizeof(*delta_weight_ih));
    if (delta_weight_ih == NULL)
    {
        perror("malloc");
        exit(EXIT_FAILURE);
    }
    init_delta_weight(delta_weight_ih, input_hidden_size);

    float* weight_ho = malloc(hidden_output_size * sizeof(*weight_ho));
    if (weight_ho == NULL)
    {
        perror("malloc");
        exit(EXIT_FAILURE);
    }
    init_weight(weight_ho, NUMOUT, NUMHID);

    float* delta_weight_ho = malloc(hidden_output_size * sizeof(*delta_weight_ho));
    if (delta_weight_ho == NULL)
    {
        perror("malloc");
        exit(EXIT_FAILURE);
    }
    init_delta_weight(delta_weight_ho, hidden_output_size);

    size_t batch_count      = NUMPAT / BSIZE;
    size_t extra_batches    = batch_count % nprocs;
    size_t batches_per_proc = batch_count / nprocs;
    struct batch_chunk* tmp = malloc(sizeof(*tmp) * nprocs);
    if (tmp == NULL)
    {
        perror("malloc");
        exit(EXIT_FAILURE);
    }

    // This array holds indexes to the start and end of the batches assigned to each process.
    struct batch_chunk* assigned_chunks = tmp;
    memset(assigned_chunks, 0, sizeof(*assigned_chunks) * nprocs);

    for (size_t i = 0; i < nprocs; i++)
    {
        assigned_chunks[i].start = batches_per_proc * i;
        assigned_chunks[i].end   = batches_per_proc * (i + 1);
    }

    if (my_rank == 0)
    {
        printf("\n\n\t---------- START OF PROGRAM OUTPUT --------------\n");
        printf("\n\nStarting training with:\n\tBatches: %lu\n\tProcesses: %d\n\tBatches per process: %lu\n\tExtra batches: %lu\n", batch_count, nprocs, batches_per_proc, extra_batches);
        printf("\nBatches assigned before load-balancing:");
        for (size_t i = 0; i < nprocs; i++)
        {
            printf("\n\tP%lu: %lu-%lu", i, assigned_chunks[i].start, assigned_chunks[i].end - 1);
        }
    }

    // Assign extra batches in a round-robin fashion, if any
    if (my_rank == 0)
    {
        printf("\n\nAssigning %lu extra batches", extra_batches);
    }
    size_t i         = 0;
    size_t remaining = extra_batches;

    while (remaining)
    {
        assigned_chunks[i].end += 1;
        for (size_t j = i + 1; j < nprocs; j++)
        {
            assigned_chunks[j].start += 1;
            assigned_chunks[j].end   += 1;
        }
        remaining--;
        i++;
    }

    if (my_rank == 0)
    {
        printf("\nFinal batch assignment:\n");
        for (size_t i = 0; i < nprocs; i++)
        {
            size_t start = assigned_chunks[i].start;
            size_t end   = assigned_chunks[i].end - 1;
            printf("\tP%lu: %lu to %lu\n", i, start, end);
            fflush(stdout);
        }
        printf("\n");
    }

    float Error; // Global accumulated error
    float BError; // Error per batch, we add it to the global error after each batch
    #pragma omp parallel num_threads(4)
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        #pragma omp single
        randomize_pattern_order(ranpat);

        Error = 0.0;
        for (int nb = assigned_chunks[my_rank].start; nb < assigned_chunks[my_rank].end; nb++) // repeat for all batches
        {
            BError = 0.0;
            for (int np = nb * BSIZE; np < (nb + 1) * BSIZE; np++) // repeat for all the training patterns within the batch
            {
                int p = ranpat[np];

                #pragma omp for
                for (int i = 0; i < numHid; i++) // compute hidden unit activations
                {
                    float SumH = 0.0;
                    for (int j = 0; j < numIn; j++)
                    {
                        SumH += f_and(weight_ih[i * NUMIN + j], tSet_msk[p * 1024 + j]);
                    }
                    Hidden[i] = 1.0 / (1.0 + exp(-SumH));
                }

                
                #pragma omp for reduction(+: BError)
                for (int k = 0; k < numOut; k++) // compute output unit activations and errors
                {
                    float SumO = 0.0;
                    for (int j = 0; j < numHid; j++)
                    {
                        SumO += Hidden[j] * weight_ho[k * NUMHID + j];
                    }
                    Output[k] = 1.0 / (1.0 + exp(-SumO));                                      // Sigmoidal Outputs
                    BError   += 0.5 * (Target[p][k] - Output[k]) * (Target[p][k] - Output[k]); // SSE
                    DeltaO[k] = (Target[p][k] - Output[k]) * Output[k] * (1.0 - Output[k]);    // Sigmoidal Outputs, SSE
                }

                #pragma omp for
                for (int j = 0; j < numHid; j++)                                               // update delta weights DeltaWeightIH
                {
                    float SumDOW = 0.0;
                    for (int k = 0; k < numOut; k++)
                    {
                        SumDOW += weight_ho[k * NUMHID + j] * DeltaO[k];
                    }
                    DeltaH[j] = SumDOW * Hidden[j] * (1.0 - Hidden[j]);

                    for (int i = 0; i < numIn; i++)
                    {
                        delta_weight_ih[j * NUMIN + i] = f_and(k_eta * DeltaH[j], tSet_msk[p * 1024 + i]) + k_alpha * delta_weight_ih[j * NUMIN + i];
                    }
                }

                #pragma omp for
                for (int k = 0; k < numOut; k++) // update delta weights DeltaWeightHO
                {
                    for (int j = 0; j < numHid; j++)
                    {
                        delta_weight_ho[k * NUMHID + j] = k_eta * Hidden[j] * DeltaO[k] + k_alpha * delta_weight_ho[k * NUMHID + j];
                    }
                }
            }

            // Update WeightIH
            #pragma omp for
            for (int i = 0; i < numHid; i++)
            {
                for (int j = 0; j < numIn; j++)
                {
                    weight_ih[i * NUMIN + j] += delta_weight_ih[i * NUMIN + j];
                }
            }

            // Update WeightHO
            #pragma omp for
            for (int i = 0; i < numOut; i++)
            {
                for (int j = 0; j < numHid; j++)
                {
                    weight_ho[i * NUMHID + j]    += delta_weight_ho[i * NUMHID + j];
                }
            }

            #pragma omp single
            Error += BError;
        }

        // Do the reductions and updates, just one thread
        #pragma omp single
        {
            MPI_Allreduce(MPI_IN_PLACE, weight_ih, numHid * numIn, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, weight_ho, numOut * numHid, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &Error, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        }

        #pragma omp for
        for (int j = 0; j < numHid; j++)
        {
            for (int i = 0; i < numIn; i++)
            {
                weight_ih[j * NUMIN + i] /= nprocs;
            }
        }

        #pragma omp for
        for (int k = 0; k < numOut; k++)
        {
            for (int j = 0; j < numHid; j++)
            {
                weight_ho[k * NUMHID + j] /= nprocs;
            }
        }

        #pragma omp single
        {
            Error /= nprocs;
            Error /= ((NUMPAT / (float) BSIZE) * BSIZE); //mean error for the last epoch

            if (my_rank == 0)
            {
                if (!(epoch % 100))
                {
                    printf("\nEpoch %-5d: Error = %f\n", epoch, Error);
                }
            }
        }

        if (Error < 0.0004)
        {

            if (my_rank == 0)
            {
                printf("\nEpochs needed: %lu", epoch);
            }
            break;
        }
    }


    for (int j = 0; j < numHid; j++)
    {
        for (int i = 0; i < numIn; i++)
        {
            WeightIH[j][i] = weight_ih[j * NUMIN + i];
        }
    }

    for (int k = 0; k < numOut; k++)
    {
        for (int j = 0; j < numHid; j++)
        {
            WeightHO[k][j] = weight_ho[k * NUMHID + j];
        }
    }

    freeTSet(NUMPAT, tSet);
    free(tSet_msk);
    MPI_Finalize();
}

int main(int argc, char** argv)
{
    // Read parameters from CLI
    const int epochs = (argc > 1) ? atoi(argv[1]) : 1000000;
    const int numIn = (argc > 2) ? atoi(argv[2]) : NUMIN;
    const int numHid = (argc > 3) ? atoi(argv[3]) : NUMHID;
    const int numOut = (argc > 4) ? atoi(argv[4]) : NUMOUT;

    // Init MPI environment
    int my_rank;
    int nprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    clock_t start = clock();

    trainN(my_rank, nprocs, epochs, argc, argv, numIn, numHid, numOut);
    if (my_rank == 0) printf("\nEND TRAINING\n");

    if (my_rank == 0) runN(numIn, numHid, numOut);

    clock_t end = clock();

    if (my_rank == 0) printf("\n\nGoodbye! (%f sec)\n\n", (end - start) / (1.0 * CLOCKS_PER_SEC));
    return 0;
}

/*******************************************************************************/
