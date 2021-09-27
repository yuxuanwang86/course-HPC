#include <iostream>
#include <chrono>
#include "omp.h"
/*
OMP_NUM_THREADS (if present) specifies initially the number of threads;
calls to omp_set_num_threads() override the value of OMP_NUM_THREADS;
the presence of the num_threads clause overrides both other values.
*/
int main(int argc, char **argv) { 
    #pragma omp parallel 
        std::cout<<"my id: "<<omp_get_thread_num()<<'\n';
    #pragma omp single 
        std::cout<<"Hello, world!"<<'\n';
    
    
    return 0;
}