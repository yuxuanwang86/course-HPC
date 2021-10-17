#include <iostream>
#include <chrono>
#include <vector>
#include "omp.h"
#define N 16
#define T 4
int main(int argc, char** argv) {
    int A[N][N];
    
    int num_blocs = N / T;

    bool flag[num_blocs+1][num_blocs+1];
    #pragma omp parallel  
    {   
        #pragma omp for collapse(2)
        for (int i = 0; i < num_blocs; i++) {
            for (int j = 0; j < num_blocs; j++) {
                #pragma omp task shared(num_blocs) firstprivate(i, j)
                for (int b_i = i * T; b_i < (i+1) * T; b_i++) {
                    for (int b_j = j * T; b_j < (j+1) * T; b_j++) {
                        A[b_i][b_j] = b_i + b_j; 
                    }
                }
            }
        }

        #pragma omp single
        for(int b_i = 0; b_i < num_blocs; b_i++) {
            for(int b_j = 0; b_j < num_blocs; b_j++) {
                #pragma omp task depend (in: flag[b_i+1][b_j], flag[b_i][b_j+1], A[b_i][b_j]) depend(out: A[b_i+1][b_j+1]) shared(num_blocs) firstprivate(b_i, b_j)
                for (int i = b_i * T; i < (b_i + 1) * T; i++)  {
                    for (int j = b_j * T; j < (b_j + 1) * T; j++) {
                        if (i == 0 && j == 0) continue;
                        else if (i == 0) {
                            A[i][j] += A[i][j - 1]; 
                        }
                        else if (j == 0) {
                            A[i][j] += A[(i-1)][j];
                        } 
                        else {
                            A[i][j] += A[(i-1)][j] + A[i][j - 1] - A[i - 1][j - 1];
                        }
                    }
                }
            }
        }
    }
    

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout<<A[i][j]<<' '; 
        }
        std::cout<<'\n';
    }

    return 0;
}