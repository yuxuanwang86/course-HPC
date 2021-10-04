#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <assert.h>
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    const int N = 1000;
    const int M = 100;
    int A[N][N];
    int B[N][N];
    
    // init A
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            A[i][j] = 1;
            B[i][j] = 0;
        }
    }
    // compute B
    
    int num_blocs = N / M;
    char flag[num_blocs][num_blocs];
    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel
    {
        #pragma omp single
        for (size_t bi = 0; bi < num_blocs; bi++) {
            for (size_t bj = 0; bj < num_blocs; bj++) {
                #pragma omp task firstprivate(bi, bj) shared(N) depend(in: flag[bi][bj], flag[bi+1][bj], flag[bi][bj+1]) depend(out: flag[bi+1][bj+1])
                for (size_t i = bi * M; i < bi * M + M; i++) {
                    for (size_t j = bj * M; j < bj * M + M; j++) {
                        if (i == 0 && j == 0) B[0][0] = A[0][0];
                        else if (i == 0) B[0][j] += B[0][j-1] + A[0][j];
                        else if (j == 0) B[i][0] += B[i-1][0] + A[i][0];
                        else B[i][j] += B[i-1][j] - B[i-1][j-1] + B[i][j-1] + A[i][j];
                    }
                }
            }
        }
    }
    std::chrono::duration<double> temps = std::chrono::high_resolution_clock::now() - start;
    std::cout << std::scientific << "Temps d'execution: " << temps.count() << "s" << std::endl;


    return 0;
}