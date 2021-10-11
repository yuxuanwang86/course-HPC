#include <iostream>
#include <chrono>
#include <vector>
#include "omp.h"
#define N 10
#define T 2

// N == 20, T == 2 -> 4 elem in each bloc
// C[0*0 + 0], C[0*0 + 1]
const int elem_per_bloc = T * T;
const int num_bloc = N / T;
void mat_mul(std::vector<int>& A, std::vector<int>& B, std::vector<int>& C, int i, int j, int k) {
    for (int b = 0; b < num_bloc; b++) {

        C[i + b * N + j] += A[i + b * N + k] * B[i + b * N + k];
           
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


    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout<<C[i * N + j]<<' ';
        }
        std::cout<<std::endl;
    }
    return 0;
}