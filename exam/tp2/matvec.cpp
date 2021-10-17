#include <iostream>
#include <chrono>
#include <vector>
#include "omp.h"
#include <immintrin.h>
#define NREPET 1024

int main(int argc, char **argv)
{
  std::cout << "Produit matrice-vecteur avec OpenMP\n";
  if (argc < 2) {
    std::cout << "Utilisation: " << argv[0] << " [num-lignes/colonnes]\n";
    std::cout << "  Example: " << argv[0] << " 1024\n";
    return 1;
  }
  int dim = std::atoi(argv[1]); 

  float* A = (float*) _mm_malloc(dim*dim*sizeof(float), 32);
  float* x = (float*) _mm_malloc(dim*sizeof(float), 32);
  float* b = (float*) _mm_malloc(dim*sizeof(float), 32);

  // Initialiser A et x tel que A(i, j) = i + j et b(j) = 1.
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      A[i * dim + j] = i + j;
    }
    x[i] = 1;
    
  }

  // Calculer b = A * x NREPET fois en sequentiel
  auto start = std::chrono::high_resolution_clock::now();
  for (int repet = 0; repet < NREPET; repet++) {
    for (int i = 0; i < dim; i++) {
      b[i] = 0;
      for (int j = 0; j < dim; j++) {
        b[i] += A[i * dim + j] * x[j];  
      }
    }
  }
  std::chrono::duration<double> tempsSeq = std::chrono::high_resolution_clock::now() - start;
  std::cout << std::scientific << "Temps d'execution sequentiel: " << tempsSeq.count() / NREPET << "s" << std::endl;

  // Calculer b = A * x NREPET fois en parallele avec omp for
  start = std::chrono::high_resolution_clock::now();
  for (int repet = 0; repet < NREPET; repet++) {
    #pragma omp parallel for
    for (int i = 0; i < dim; i+=1) {
      b[i] = 0;
      __m256 res_tmp = _mm256_set1_ps(0.0f);
      for (int j = 0; j < dim; j+=8) {
        res_tmp = _mm256_fmadd_ps(_mm256_load_ps(A + i * dim + j), _mm256_load_ps(x + j), res_tmp);
        // b[i] += A[i * dim + j] * x[j];  
      }
      float res_tab[8] __attribute__ ((aligned(32)));
      _mm256_store_ps(res_tab, res_tmp);
      for (int k = 0; k < 8; k++) b[i] += res_tab[k];
    }
  }
  std::chrono::duration<double> tempsPar = std::chrono::high_resolution_clock::now() - start;
  std::cout << std::scientific << "Temps d'execution parallele avec omp for: " << tempsPar.count() / NREPET << "s" <<
    std::endl;

  // Calculer b = A * x NREPET fois en parallele avec omp task
  start = std::chrono::high_resolution_clock::now();
  for (int repet = 0; repet < NREPET; repet++) {
    #pragma omp parallel
    {
      #pragma omp single nowait
      for (int i = 0; i < dim; i++) {
        b[i] = 0;
        #pragma omp task 
        {
          __m256 res_tmp = _mm256_set1_ps(0.0f);
          for (int j = 0; j < dim; j+=8) res_tmp = _mm256_fmadd_ps(_mm256_load_ps(A + i * dim + j), _mm256_load_ps(x + j), res_tmp);
          float res_tab[8] __attribute__ ((aligned(32)));
          _mm256_store_ps(res_tab, res_tmp);
          for (int k = 0; k < 8; k++) b[i] += res_tab[k];
        }
      }
    }
  }
    
  std::chrono::duration<double> tempsParTasks = std::chrono::high_resolution_clock::now() - start;
  std::cout << std::scientific << "Temps d'execution parallele avec omp tasks: " << tempsParTasks.count() / NREPET <<
    "s" << std::endl;

  // Calculer et afficher l'acceleration et l'efficacite de la parallelisation avec omp for
  double acceleration = 0.0;
  double efficacite = 0.0;
  // A FAIRE ...
  std::cout << "Acceleration: " << acceleration << std::endl;
  std::cout << "Efficacite: " << acceleration << std::endl;

  // Verifier le resultat. b(i) est cense etre (dim - 1) * dim / 2 + i * dim
  for (int i = 0; i < dim; i++) {
    double val = (dim - 1) * (double)dim / 2.0 + (double)i * dim;
    if (b[i] != val) {
      std::cout << "valeur incorrecte: b[" << i << "] = " << b[i] << " != " << val << std::endl;
      break;
    }
    if (i == dim - 1) {
      std::cout << "Produit matrice-vecteur est effectue avec succes!\n";
    }
  }
  _mm_free(A);
  _mm_free(b);
  _mm_free(x);
  return 0;
}
