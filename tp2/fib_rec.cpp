#include <iostream>
#include <chrono>
#include <vector>
#include "omp.h"

int fib(int n) {
    if (n < 2) {
        return n;
    }
    int i, j;
    #pragma omp task shared(i)
    i = fib(n - 1);
    #pragma omp task shared(j)
    j = fib(n - 2);
    #pragma omp taskwait
    return i + j;
}

int main(int argc, char **argv) {
    int n = 10;
    #pragma omp parallel
    #pragma omp single
    std::cout << "fib(n): "<< fib(n) << std::endl;
}