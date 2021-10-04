#include <iostream>
#include <chrono>
#include <vector>
#include "omp.h"

int main(int argc, char **argv) {
    const int n = 11;
    int fib[11]; 
    fib[0] = 0;
    fib[1] = 1;
    #pragma omp parallel
    #pragma omp single
    for (size_t i = 2; i < n; i++) {
        #pragma omp task depend(in: fib[i-1], fib[i-2]) depend(out: fib[i])
        fib[i] = fib[i-1] + fib[i-2];
    }

    std::cout << fib[n-1] << std::endl;
    return 0;
}