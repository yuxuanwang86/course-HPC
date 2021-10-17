#include <chrono>
#include <iostream>
#include "omp.h"

inline double f(double x)
{
  return (4 / (1 + x * x));
}

int main()
{
  int i;
  const int P = 4;
  const int N = 100000000;
  double pi = 0.0;
  double s = double(1) / double(N);
  // Calculer le pi en sequentiel
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < N; i++) pi += s * (f(i * s) + f((i + 1) * s)) / double(2);
  std::cout << "pi = " << pi << std::endl;
  std::chrono::duration<double> tempsSeq = std::chrono::high_resolution_clock::now() - start;
  std::cout << "Temps sequentiel: " << tempsSeq.count() << "s\n";

  // Calculer le pi avec omp for et reduction
  pi = 0.0;
  start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel for reduction(+:pi)
  for (int i = 0; i < N; i++) pi += s * (f(i * s) + f((i + 1) * s)) / double(2);
  
  std::cout << "pi = " << pi << std::endl;
  std::chrono::duration<double> tempsOmpFor = std::chrono::high_resolution_clock::now() - start;
  std::cout << "Temps parallel omp for: " << tempsOmpFor.count() << "s\n";

  // Calculer le pi avec boucle parallele faite a la main
  pi = 0.0;
  double pi_local = 0.0;
  start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel num_threads(P)
  {
    int pid = omp_get_thread_num();
    int num_per_thread = N/P;
    int start = pid * num_per_thread;
    int end = N;
    if (pid != P - 1) end = (pid + 1) * num_per_thread;
    for (int i = start; i < end; i++) pi_local += s * (f(i * s) + f((i + 1) * s)) / double(2);
    #pragma omp atomic
    pi += pi_local;
  }
  std::cout << "pi = " << pi << std::endl;
  std::chrono::duration<double> tempsForMain = std::chrono::high_resolution_clock::now() - start;
  std::cout << "Temps parallel for a la main: " << tempsForMain.count() << "s\n";

  return 0;
}
