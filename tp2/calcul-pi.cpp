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
  const int N = 100000000;
  double pi = 0.0;
  const double s = 1/double(N);
  // Calculer le pi en sequentiel
  auto start = std::chrono::high_resolution_clock::now();
  for (i = 0; i < N; i++) pi += s * (f(double(i)*s) + f(double(i+1)*s)) / 2;
  std::cout << "pi = " << pi << std::endl;
  std::chrono::duration<double> tempsSeq = std::chrono::high_resolution_clock::now() - start;
  std::cout << "Temps sequentiel: " << tempsSeq.count() << "s\n";
  pi = 0.0;
  // Calculer le pi avec omp for et reduction
  start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel for num_threads(10) reduction(+: pi)
  for (i = 0; i < N; i++) pi += (s * (f(double(i)*s) + f(double(i+1)*s)) / 2);
  
  std::cout << "pi = " << pi << std::endl;
  std::chrono::duration<double> tempsOmpFor = std::chrono::high_resolution_clock::now() - start;
  std::cout << "Temps parallel omp for: " << tempsOmpFor.count() << "s\n";

  // Calculer le pi avec boucle parallele faite a la main
  pi = 0.0;
  start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel num_threads(10)
  {
    int tid = omp_get_thread_num();
    int prop = N / omp_get_num_threads();
    double local_pi = 0.0;
    for (i = tid * prop; i < tid * prop + prop; i++) local_pi += (s * (f(double(i)*s) + f(double(i+1)*s)) / 2);
    #pragma omp atomic
    pi += local_pi;
  }
  std::cout << "pi = " << pi << std::endl;
  std::chrono::duration<double> tempsForMain = std::chrono::high_resolution_clock::now() - start;
  std::cout << "Temps parallel for a la main: " << tempsForMain.count() << "s\n";

  return 0;
}
