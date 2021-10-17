#include <iostream>
#include <chrono>
#include <vector>
#include "omp.h"
#include <immintrin.h>
#define N 4096
#define T 32



const int num_bloc = N / T;
void mat_mul(float* A, float* B, float* C, int b_i, int b_j/* , int b_k */) {
    for (int i = 0; i < T; i++) {
        for (int j = 0; j < T; j++) {
            // for (int k = 0; k < T; k++) {
                __m256 res_tmp1 = _mm256_set1_ps(0.0f);
                __m256 res_tmp2 = _mm256_set1_ps(0.0f);
                __m256 res_tmp3 = _mm256_set1_ps(0.0f);
                __m256 res_tmp4 = _mm256_set1_ps(0.0f);
                res_tmp1 = _mm256_fmadd_ps(_mm256_load_ps(A + i *  N + b_i * T * N), _mm256_load_ps(B + j *  N + b_j * T * N), res_tmp1);
                res_tmp2 = _mm256_fmadd_ps(_mm256_load_ps(A + i *  N + b_i * T * N), _mm256_load_ps(B + j *  N + b_j * T * N + 8), res_tmp2);
                res_tmp3 = _mm256_fmadd_ps(_mm256_load_ps(A + i *  N + b_i * T * N), _mm256_load_ps(B + j *  N + b_j * T * N + 16), res_tmp3);
                res_tmp4 = _mm256_fmadd_ps(_mm256_load_ps(A + i *  N + b_i * T * N), _mm256_load_ps(B + j *  N + b_j * T * N + 24), res_tmp4);
                float res_tab[8] __attribute__ ((aligned(32)));
                res_tmp1 = _mm256_add_ps(res_tmp1, res_tmp2);
                res_tmp3 = _mm256_add_ps(res_tmp3, res_tmp4);
                res_tmp1 = _mm256_add_ps(res_tmp1, res_tmp3);
                _mm256_store_ps(res_tab, res_tmp1);
                for (int i = 0; i < 8; i++) C[i *  N + b_i * T * N + b_j * T + j] += res_tab[i];
                // C[i *  N + b_i * T * N + b_j * T + j] += A[i *  N + b_i * T * N  + b_k * T + k] * B[j *  N + b_j * T * N + b_k * T + k];
            // }
        }
    }
}

int main(int argc, char **argv)
{
    float* A = (float*) _mm_malloc(N * N * sizeof(float), 32);
    float* B_T = (float*) _mm_malloc(N * N * sizeof(float), 32);
    float* C = (float*) _mm_malloc(N * N * sizeof(float), 32);

    for (int i = 0; i < N*N; i++) {
        A[i] = i;
        B_T[i] = i + 1;
        C[i] = 0;
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int b_i = 0; b_i < num_bloc; b_i++) {
        for (int b_j = 0; b_j < num_bloc; b_j++) {
            // for (int b_k = 0; b_k < num_bloc; b_k++)
                mat_mul(A, B_T, C, b_i, b_j /*, b_k */);
        }
    }
    
    
    
    
    std::chrono::duration<double> temps = std::chrono::high_resolution_clock::now() - start;
    std::cout << "Temps: " << temps.count() << "s\n";


    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         std::cout<<C[i * N + j]<<'\t';
    //     }
    //     std::cout<<std::endl;
    // }
    _mm_free(C);
    _mm_free(B_T);
    _mm_free(A);
    return 0;
}