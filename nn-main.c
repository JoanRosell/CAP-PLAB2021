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
//#include <mpi/mpi.h>
#include <mpi.h>
#include "common.h"

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

//void trainN(const int epochs, const int numIn, const int numHid, const int numOut)
//{
//    char** tSet;
//
//    float DeltaWeightIH[numHid][numIn], DeltaWeightHO[numOut][numHid];
//    float Error, BError, eta = 0.3, alpha = 0.5, smallwt = 0.22;
//    int   ranpat[NUMPAT];
//    float Hidden[numHid], Output[numOut], DeltaO[numOut], DeltaH[numHid];
//    float SumO, SumH, SumDOW;
//    float inv_WeightHO[NUMHID][NUMOUT];
//
//    if ((tSet = loadPatternSet(NUMPAT, "optdigits.tra", 1)) == NULL)
//    {
//        printf("Loading Patterns: Error!!\n");
//        exit(-1);
//    }
//
//    uint32_t* tSet_msk = malloc(sizeof(uint32_t) * NUMPAT * 1024);
//
//    for (size_t i = 0; i < NUMPAT; i++)
//    {
//        for (size_t j = 0; j < 1024; j++)
//        {
//            tSet_msk[i * 1024 + j] = tSet[i][j] * 0xFFFFFFFF;
//        }
//    }
//
//    for (int i = 0; i < numHid; i++)
//    {
//        for (int j = 0; j < numIn; j++)
//        {
//            WeightIH[i][j]      = 2.0 * (frando() + 0.01) * smallwt;
//            DeltaWeightIH[i][j] = 0.0;
//        }
//    }
//
//    for (int i = 0; i < numOut; i++)
//    {
//        for (int j = 0; j < numHid; j++)
//        {
//            WeightHO[i][j]      = 2.0 * (frando() + 0.01) * smallwt;
//            DeltaWeightHO[i][j] = 0.0;
//            inv_WeightHO[j][i]  = WeightHO[i][j];
//        }
//    }
//
//    Error = 10;
//    for (int epoch = 0; epoch < epochs && Error >= 0.0004; epoch++) // iterate weight updates
//    {
//        {
//            for (int p = 0; p < NUMPAT; p++) // randomize order of individuals
//            {
//                ranpat[p] = p;
//            }
//            for (int p = 0; p < NUMPAT; p++)
//            {
//                int x  = rando();
//                int np = (x * x) % NUMPAT;
//                int op = ranpat[p];
//                ranpat[p]  = ranpat[np];
//                ranpat[np] = op;
//            }
//
//            printf(".");
//            fflush(stdout);
//        }
//
//
//        size_t batch_count   = NUMPAT / BSIZE;
//        size_t extra_batches = batch_count % nproc;
//
//        Error = 0.0;
//        for (int nb = 0; nb < NUMPAT / BSIZE; nb++) // repeat for all batches
//        {
//            BError = 0.0;
//            for (int np = nb * BSIZE; np < (nb + 1) * BSIZE; np++) // repeat for all the training patterns within the batch
//            {
//                int p = ranpat[np];
//
//                for (int j = 0; j < numHid; j++) // compute hidden unit activations
//                {
//                    float SumH = 0.0;
//                    for (int i = 0; i < numIn; i++)
//                    {
//                        SumH += f_and(WeightIH[j][i], tSet_msk[p * 1024 + i]);
//                    }
//                    Hidden[j] = 1.0 / (1.0 + exp(-SumH));
//                }
//
//                for (int k = 0; k < numOut; k++) // compute output unit activations and errors
//                {
//                    float SumO = 0.0;
//                    for (int j = 0; j < numHid; j++)
//                    {
//                        SumO += Hidden[j] * WeightHO[k][j];
//                    }
//                    Output[k] = 1.0 / (1.0 + exp(-SumO));                                      // Sigmoidal Outputs
//                    BError   += 0.5 * (Target[p][k] - Output[k]) * (Target[p][k] - Output[k]); // SSE
//                    DeltaO[k] = (Target[p][k] - Output[k]) * Output[k] * (1.0 - Output[k]);    // Sigmoidal Outputs, SSE
//                }
//
//                for (int j = 0; j < numHid; j++)                                               // update delta weights DeltaWeightIH
//                {
//                    float SumDOW = 0.0;
//                    for (int k = 0; k < numOut; k++)
//                    {
//                        SumDOW += inv_WeightHO[j][k] * DeltaO[k];
//                    }
//                    DeltaH[j] = SumDOW * Hidden[j] * (1.0 - Hidden[j]);
//                    for (int i = 0; i < numIn; i++)
//                    {
//                        DeltaWeightIH[j][i] = f_and(eta * DeltaH[j], tSet_msk[p * 1024 + i]) + alpha * DeltaWeightIH[j][i];
//                    }
//                }
//
//                for (int k = 0; k < numOut; k++) // update delta weights DeltaWeightHO
//                {
//                    for (int j = 0; j < numHid; j++)
//                    {
//                        DeltaWeightHO[k][j] = eta * Hidden[j] * DeltaO[k] + alpha * DeltaWeightHO[k][j];
//                    }
//                }
//            }
//
//            for (int j = 0; j < numHid; j++) // update weights WeightIH
//            {
//                for (int i = 0; i < numIn; i++)
//                {
//                    WeightIH[j][i] += DeltaWeightIH[j][i];
//                }
//            }
//
//            for (int k = 0; k < numOut; k++) // update weights WeightHO
//            {
//                for (int j = 0; j < numHid; j++)
//                {
//                    WeightHO[k][j]    += DeltaWeightHO[k][j];
//                    inv_WeightHO[j][k] = WeightHO[k][j];
//                }
//            }
//
//            Error += BError; // We only want to update Error once per iteration
//        }
//
//
//        {
//            Error = Error / ((NUMPAT / BSIZE) * BSIZE); //mean error for the last epoch
//            if (!(epoch % 100))
//            {
//                printf("\nEpoch %-5d :   Error = %f \n", epoch, Error);
//            }
//            if (Error < 0.0004)
//            {
//                printf("\nEpoch %-5d :   Error = %f \n", epoch, Error);
//            }
//        }
//    }
//
//    freeTSet(NUMPAT, tSet);
//    free(tSet_msk);
//    printf("END TRAINING\n");
//}

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

struct batch_chunk
{
    size_t start;
    size_t end;
};

int main(int argc, char** argv)
{
    // Read parameters from CLI
    const int epochs = (argc > 1) ? atoi(argv[1]) : 1000000;
    const int numIn = (argc > 2) ? atoi(argv[2]) : NUMIN;
    const int numHid = (argc > 3) ? atoi(argv[3]) : NUMHID;
    const int numOut = (argc > 4) ? atoi(argv[4]) : NUMOUT;
    int       my_rank, nprocs;
    //MPI_Status status;

    clock_t start = clock();

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // trainN(epochs, numIn, numHid, numOut);
    char** tSet;

    float DeltaWeightIH[numHid][numIn], DeltaWeightHO[numOut][numHid];
    float Error, BError, eta = 0.3, alpha = 0.5, smallwt = 0.22;
    int   ranpat[NUMPAT];
    float Hidden[numHid], Output[numOut], DeltaO[numOut], DeltaH[numHid];
    float inv_WeightHO[NUMHID][NUMOUT];

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

    Error = 10;
    for (int epoch = 0; epoch < epochs; epoch++) // iterate weight updates
    {
        for (int p = 0; p < NUMPAT; p++)
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

        Error = 0.0;
        for (int nb = assigned_chunks[my_rank].start; nb < assigned_chunks[my_rank].end; nb++) // repeat for all batches
        {
            BError = 0.0;
            for (int np = nb * BSIZE; np < (nb + 1) * BSIZE; np++) // repeat for all the training patterns within the batch
            {
                int p = ranpat[np];

                for (int j = 0; j < numHid; j++) // compute hidden unit activations
                {
                    float SumH = 0.0;
                    for (int i = 0; i < numIn; i++)
                    {
                        SumH += f_and(WeightIH[j][i], tSet_msk[p * 1024 + i]);
                    }
                    Hidden[j] = 1.0 / (1.0 + exp(-SumH));
                }

                for (int k = 0; k < numOut; k++) // compute output unit activations and errors
                {
                    float SumO = 0.0;
                    for (int j = 0; j < numHid; j++)
                    {
                        SumO += Hidden[j] * WeightHO[k][j];
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
                        SumDOW += inv_WeightHO[j][k] * DeltaO[k];
                    }
                    DeltaH[j] = SumDOW * Hidden[j] * (1.0 - Hidden[j]);
                    for (int i = 0; i < numIn; i++)
                    {
                        DeltaWeightIH[j][i] = f_and(eta * DeltaH[j], tSet_msk[p * 1024 + i]) + alpha * DeltaWeightIH[j][i];
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

            // Update WeightIH
            for (int j = 0; j < numHid; j++)
            {
                for (int i = 0; i < numIn; i++)
                {
                    WeightIH[j][i] += DeltaWeightIH[j][i];
                }
            }

            // Update WeightHO
            for (int k = 0; k < numOut; k++)
            {
                for (int j = 0; j < numHid; j++)
                {
                    WeightHO[k][j]    += DeltaWeightHO[k][j];
                    inv_WeightHO[j][k] = WeightHO[k][j];
                }
            }

            Error += BError;
        }

        MPI_Allreduce(MPI_IN_PLACE, WeightIH, numHid * numIn, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, WeightHO, numOut * numHid, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        MPI_Allreduce(MPI_IN_PLACE, &Error, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        Error /= nprocs;

        for (int j = 0; j < numHid; j++)
        {
            for (int i = 0; i < numIn; i++)
            {
                WeightIH[j][i] /= nprocs;
            }
        }

        for (int k = 0; k < numOut; k++)
        {
            for (int j = 0; j < numHid; j++)
            {
                WeightHO[k][j] /= nprocs;
            }
        }

        Error /= ((NUMPAT / (float) BSIZE) * BSIZE); //mean error for the last epoch
        if (Error < 0.0004)
        {
            break;
        }

        if (my_rank == 0)
        {
            if (!(epoch % 100))
            {
                printf("\nEpoch %-5d: Error = %f\n", epoch, Error);
            }
        }
    }

    freeTSet(NUMPAT, tSet);
    free(tSet_msk);
    MPI_Finalize();

    printf("END TRAINING\n");

    runN(numIn, numHid, numOut);

    clock_t end = clock();

    printf("\n\nGoodbye! (%f sec)\n\n", (end - start) / (1.0 * CLOCKS_PER_SEC));

    return 0;
}

/*******************************************************************************/
