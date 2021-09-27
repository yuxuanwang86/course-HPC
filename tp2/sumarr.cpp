#include <iostream>
#include <chrono>
#include "omp.h"
#include <vector>
#define N 10
int main(int argc, char **argv) { 
    std::vector<float> A(N);
    float sum = 0.0f;
    #pragma omp parallel 
    {
        #pragma omp for
        for (size_t i = 0; i < N; i++) A[i] = i;
        #pragma omp for reduction(+: sum)
        for (size_t i = 0; i < N; i++) sum += A[i];
    }
    
    
    // for (size_t i = 0; i < N; i++) std::cout<<A[i]<<'\n';
    std::cout<<sum<<'\n';
    return 0;
}