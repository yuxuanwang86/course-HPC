#include <iostream>
#include <chrono>
#include <vector>
#include "omp.h"
#define N 4096
#define T 32

// N == 20, T == 2 -> 4 elem in each bloc
// C[0*0 + 0], C[0*0 + 1]

const int num_bloc = N / T;
void mat_mul(std::vector<int>& A, std::vector<int>& B, std::vector<int>& C, int i, int j, int k) {
    for (int b_i = 0; b_i < num_bloc; b_i++) {
        for (int b_j = 0; b_j < num_bloc; b_j++)
            for (int b_k = 0; b_k < num_bloc; b_k++)
                C[i * num_bloc * N + j * num_bloc + b_i * N + b_j] += A[i * num_bloc * N + k * num_bloc + b_i * N + b_k] * B[j * num_bloc * N + k * num_bloc + b_j * N + b_k];    
    }
}

int main(int argc, char **argv)
{
    
    std::vector<int> A(N * N);
    std::vector<int> B_T(N * N);
    std::vector<int> C(N * N);
    for (int i = 0; i < N*N; i++) {
        A[i] = B_T[i] = 1;
        C[i] = 0;
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < T; i++) {
        for (int j = 0; j < T; j++) {
            for (int k = 0; k < T; k++)
                mat_mul(A, B_T, C, i, j, k);
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
    return 0;
}