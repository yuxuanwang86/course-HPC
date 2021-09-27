/**
  * Copie d'un tableau dans un autre avec les intrinseques AVX.
  * A compiler avec les drapeaux -O2 -mavx2.
  */

#include <immintrin.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <algorithm>
#define NREPET 1

void afficherUsage()
{ 
  printf("Usage: ./inversion-tableau-avx [taille-du-tableau]\n");
}

int main(int argc, char **argv)
{
  if (argc < 2) { 
    afficherUsage();
    return 1;
  }
  int dim = std::atoi(argv[1]);
  
  // Allouer et initialiser tableau tab de taille dim aligne par 32 octets
  float *tab = (float*) _mm_malloc (dim * sizeof(float), 32);;
  for (auto i = 0; i < dim; i++) tab[i] = i;
  //for (size_t i = dim-1; i >= dim - 20; i--) std::cout<< tab[i] << ' ';
  std::cout<< '\n';
  // Inverser le tableau en place~(c'est a dire sans utiliser un deuxieme tableau auxiliaire) sans vectorisation
  // On repete NREPET fois pour mieux mesurer le temps d'execution
  auto start = std::chrono::high_resolution_clock::now();
  //for (int repet = 0; repet < NREPET; repet++) {
  //  std::reverse(tab, tab + dim);
  //}
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diffSeq = end-start;
  std::cout << std::scientific << "Inversion sans AVX: " << diffSeq.count() / NREPET << "s" << std::endl;

  // Inverser le tableau en place avec AVX~(c'est a dire sans utiliser un deuxieme tableau auxiliaire)
  // On repete NREPET fois pour mieux mesurer le temps d'execution
  start = std::chrono::high_resolution_clock::now();
  for (int repet = 0; repet < NREPET; repet++) {
    __m256i perm_idx = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    size_t i = 0;
    size_t j = dim - 8;
    for (; i <= j; i += 8, j -= 8) {
      __m256 left = _mm256_permutevar8x32_ps(_mm256_load_ps(tab+i), perm_idx);
      __m256 right = _mm256_permutevar8x32_ps(_mm256_load_ps(tab+j), perm_idx);
      _mm256_store_ps(tab+j, left);
      _mm256_store_ps(tab+i, right);
    }
  }
  end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diffPar = end-start;
  std::cout << std::scientific << "Inversion avec AVX: " << diffPar.count() / NREPET << "s" << std::endl;
  // Afficher l'acceleration et l'efficacite
  double acceleration = diffSeq.count() / diffPar.count();
  double efficacite = acceleration / 8;
  std::cout << std::fixed << std::setprecision(2) << "Acceleration: " << acceleration << std::endl;
  std::cout << "Efficacite: " << 100 * efficacite << "%" << std::endl;
  for (size_t i = 0; i < 20; i++) std::cout<< tab[i] << ' ';  
  // Desallouer le tableau tab
  _mm_free(tab);

 

  return 0;
}
