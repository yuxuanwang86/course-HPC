#include <iostream>
#include <chrono>
#include <vector>
#include "omp.h"

int fib_rec(int n) {
    if (n < 2) return n;
    int x, y;
    #pragma omp task shared(x, y)
    {
        x = fib_rec(n - 1);
        y = fib_rec(n - 2);
    }

    #pragma omp taskwait
    return x + y;
}

int main(int argc, char** argv) {
    // #pragma omp parallel
    // #pragma omp single
    // std::cout<<"fib(5): "<<fib_rec(5)<<'\n';
    const int N = 10;
    int arr[N];
    arr[0] = 0;
    arr[1] = 1;
    #pragma omp parallel
    #pragma omp single
    for (int i = 2; i < N; i++) {
        #pragma omp task depend(in: arr[i-1], arr[i-2]) depend(out: arr[i])
        arr[i] = arr[i-1] + arr[i-2];
    }
    std::cout<<"fib(5): "<<arr[5]<<'\n';
    return 0;
}