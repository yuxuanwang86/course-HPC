#include <iostream>
#include <chrono>
#include <vector>
#include "omp.h"
#define N 8
#define T 4



const int num_bloc = N / T;
void mat_mul(std::vector<int>& A, std::vector<int>& B, std::vector<int>& C, int b_i, int b_j, int b_k) {
    
    #pragma omp for collapse(3)
    for (int i = 0; i < T; i++) {
        for (int j = 0; j < T; j++) {
             for (int k = 0; k < T; k++) {
                #pragma omp reduction(+: C[i *  N + b_i * T * N + b_j * T + j])
                C[i *  N + b_i * T * N + b_j * T + j] += A[i *  N + b_i * T * N  + b_k * T + k] * B[j *  N + b_j * T * N + b_k * T + k];    
            }
        }
    }
    
}

int main(int argc, char **argv)
{
    
    std::vector<int> A(N * N);
    std::vector<int> B_T(N * N);
    std::vector<int> C(N * N);
    for (int i = 0; i < N*N; i++) {
        A[i] = i;
        B_T[i] = i + 1;
        C[i] = 0;
    }

    auto start = std::chrono::high_resolution_clock::now();
    std::cout<<"yo";
    #pragma omp parallel shared(A, B_T, C, num_bloc)
    {
        #pragma omp single 
        for (int b_i = 0; b_i < num_bloc; b_i++) {
            for (int b_j = 0; b_j < num_bloc; b_j++) {
                for (int b_k = 0; b_k < num_bloc; b_k++) { mat_mul(A, B_T, C, b_i, b_j, b_k); }
            }
        }
    } 
    
    
    
    
    
    std::chrono::duration<double> temps = std::chrono::high_resolution_clock::now() - start;
    std::cout << "Temps: " << temps.count() << "s\n";


    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout<<C[i * N + j]<<'\t';
        }
        std::cout<<std::endl;
    }
    return 0;
}