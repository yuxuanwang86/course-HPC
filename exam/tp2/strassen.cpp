#include <iostream>
#include <chrono>
#include <vector>
#include "omp.h"
#include <immintrin.h>
#define N 4096
#define T 32

void matmul(std::vector<std::vector<int>>& A, std::vector<std::vector<int>>& B, std::vector<std::vector<int>>& C, int b_i, int b_j, int b_k) {
    for(int i = 0; i < T; i++) {
        for(int j = 0; j < T; j++) {
            for(int k = 0; k < T; k++) {
                C[b_i * T + i][b_j * T + j] += A[b_i * T + i][b_k * T + k] * B[b_k * T + k][b_j * T + j];
            }  
        }  
    }    
}

void display(std::vector<std::vector<int>>& C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout<<C[i][j]<<' ';
        }
        std::cout<<'\n';
    }
    std::cout<<'\n';
}

int main() {
    std::vector<std::vector<int>> A(N, std::vector<int>(N));
    std::vector<std::vector<int>> B(N, std::vector<int>(N));
    std::vector<std::vector<int>> C(N, std::vector<int>(N));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = i + j;
            B[i][j] =  i + j + 1;
            C[i][j] = 0;
        }
    } 

    int num_bloc = N / T;
    
    // for(int i = 0; i < N; i++) {
    //     for(int j = 0; j < N; j++) {
    //         for(int k = 0; k < N; k++) {
    //             C[i][j] += A[i][k] * B[k][j];
    //         }  
    //     }  
    // }    
    // display(C);
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         C[i][j] = 0;
    //     }
    // } 
    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel shared(num_bloc)
    {   
        #pragma omp for collapse(3)
        for(int b_i = 0; b_i < num_bloc; b_i++) {
            for(int b_j = 0; b_j < num_bloc; b_j++) {
                for(int b_k = 0; b_k < num_bloc; b_k++) {
                    #pragma omp task firstprivate(b_i, b_j, b_k)
                    matmul(A, B, C, b_i, b_j, b_k);
                }  
            }  
        }    
    }
    
    std::chrono::duration<double> temps = std::chrono::high_resolution_clock::now() - start;
    std::cout << std::scientific << "Temps d'execution: " << temps.count() << "s" << std::endl;
    // display(C);

    return  0;
}