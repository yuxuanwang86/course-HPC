#include <chrono>
#include <iostream>
#include <vector>
#include "omp.h"

int main()
{
  int i;
  int N = 1000000000;
  std::vector<double> A(N);
  double somme = 0.0;

  auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel
  {
#pragma omp for
    for (i = 0; i < N; i++) {
      A[i] = (double)i;
    }
#pragma omp for reduction(+:somme)
    for (i = 0; i < N; i++) {
      somme += A[i];
    }
  }
  std::cout << "La somme est " << somme << std::endl;
  std::chrono::duration<double> temps = std::chrono::high_resolution_clock::now() - start;
  std::cout << "Temps de calcul: " << temps.count() << "s\n";

  return 0;
}
