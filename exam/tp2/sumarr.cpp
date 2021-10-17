#include <chrono>
#include <iostream>
#include "omp.h"
#include <vector>
#define N 64
int main(int argc, char** argv) {
    std::vector<int> A(N);
    int res = 0;
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < N; i++) { A[i] = i; }
        // #pragma omp for reduction(+:res)
        // for (int i = 0; i < N; i++) { res += A[i]; }
        #pragma omp sections reduction(+:res)
        {
            #pragma omp section
            for (int i = 0; i < N / 4; i++) res += A[i];
            #pragma omp section
            for (int i = N / 4; i < N / 2; i++) res += A[i];
            #pragma omp section
            for (int i = N / 2; i < 3 * N / 4; i++) res += A[i];
            #pragma omp section
            for (int i = 3 * N / 4; i < N; i++) res += A[i];
        }
        
    }
    std::cout<<res<<'\n';   
    return 0;
}