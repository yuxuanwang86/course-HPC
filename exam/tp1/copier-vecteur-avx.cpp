/**
  * Copie d'un tableau dans un autre avec les intrinseques AVX.
  * A compiler avec les drapeaux -O2 -mavx2.
  */

#include <immintrin.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>

#define NREPET 1024

void afficherUsage()
{ 
  printf("Usage: ./copier-vecteur-avx [taille-du-tableau]\n");
}

int main(int argc, char **argv)
{
  if (argc < 2) { 
    afficherUsage();
    return 1;
  }
  int dim = std::atoi(argv[1]);

  // Allouer et initialiser deux tableaux de flottants de taille dim alignes par 32 octets
  float* tab0 = (float*) _mm_malloc(dim * sizeof(float), 32);
  float* tab1 = (float*) _mm_malloc(dim * sizeof(float), 32);
  
  for (int i = 0; i < dim; i++) {
    tab0[i] = i;
    tab1[i] = 0;
  }
  // Copier tab0 dans tab1 de manière scalaire~(code non-vectorise).
  // On repete NREPET fois pour mieux mesurer le temps d'execution
  auto start = std::chrono::high_resolution_clock::now();
  for (int repet = 0; repet < NREPET; repet++) {
      for (int i = 0; i < dim; i++) {
        tab0[i] = tab1[i];
      }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diffSeq = end-start;
  std::cout << std::scientific << "Copier sans AVX: " << diffSeq.count() / NREPET << "s" << std::endl;
  for (int i = 0; i < dim; i++) {
    tab0[i] = i;
    tab1[i] = 0;
  }
  // Copier tab0 dans tab1 de maniere vectorisee avec AVX
  start = std::chrono::high_resolution_clock::now();
  for (int repet = 0; repet < NREPET; repet++) {
    int i;
    for (i = 0; i < dim - 7; i += 8) {
      _mm256_store_ps(tab1 + i, _mm256_load_ps(tab0 + i));
    }
    for ( ; i< dim; i++) tab1[i] = tab0[i];
  }
  end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diffPar = end-start;
  std::cout << std::scientific << "Copier avec AVX: " << diffPar.count() / NREPET << "s" << std::endl;
  // Afficher l'acceleration et l'efficacite
  double acceleration = diffSeq.count() / diffPar.count();
  double efficacite = acceleration / 8;
  std::cout << std::fixed << std::setprecision(2) << "Acceleration: " << acceleration << std::endl;
  std::cout << "Efficacite: " << 100 * efficacite << "%" << std::endl;
  for (int i = 0; i < dim; i++) {
    tab0[i] = i;
    tab1[i] = 0;
  }
  // Copier tab0 dans tab1 de maniere vectorisee avec AVX et deroulement de facteur 4
  start = std::chrono::high_resolution_clock::now();
  for (int repet = 0; repet < NREPET; repet++) {
    int i;
    for (i = 0; i < dim - 31; i += 32) {
      _mm256_store_ps(tab1 + i, _mm256_load_ps(tab0 + i));
      _mm256_store_ps(tab1 + i + 8, _mm256_load_ps(tab0 + i + 8));
      _mm256_store_ps(tab1 + i + 16, _mm256_load_ps(tab0 + i + 16));
      _mm256_store_ps(tab1 + i + 24, _mm256_load_ps(tab0 + i + 24));
    }
    for ( ; i< dim; i++) tab1[i] = tab0[i];
  }
  end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diffParDeroule = end-start;
  std::cout << std::scientific << "Copier avec AVX et deroulement: " << diffParDeroule.count() / NREPET << "s" << std::endl;
  // Afficher l'acceleration et l'efficacite
  // A FAIRE ...

  // Desallouer les tableaux tab0 et tab1
  _mm_free(tab0);
  _mm_free(tab1);

  return 0;
}
