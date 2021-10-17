#include <chrono>
#include <iostream>
#include "omp.h"

int main(int argc, char **argv) { 
    omp_set_num_threads(4);
    #pragma omp parallel num_threads(3)
    {
        std::cout<<"my id: "<<omp_get_thread_num()<<'\n';
    #pragma omp single 
        std::cout<<"Hello, world!"<<'\n';
    }

    
    
    return 0;
}