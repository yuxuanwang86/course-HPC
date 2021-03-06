/**
  * Produit scalaire de deux tableaux avec les intrinseques AVX et FMA.
  * A compiler avec les drapeaux -O2 -mavx2 -mfma
  */

#include <immintrin.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>

#define NREPET 1024

int main(int argc, char **argv)
{
  if (argc < 2) {
    std::cout << "Utilisation: \n  " << argv[0] << " [taille-de-tableau]\n";
    return 1;
  }
  int dim = std::atoi(argv[1]);
  if (dim % 8) {
    std::cout << "La taille de tableau doit etre un multiple de 8.\n";
    return 1;
  }

  // Allouer les tableaux tab0 et tab1 de flottants de taille dim alignes par 32 octets, puis initialiser tab0[i]=i
  float* tab0 = (float*)_mm_malloc(dim * sizeof(float), 32);
  float* tab1 = (float*)_mm_malloc(dim * sizeof(float), 32);
  
  for (int i = 0; i < dim; i++) {
    tab0[i] = 1;
    tab1[i] = 1;
  }
  float res = 0.0f;
  // Faire le produit scalaire non-vectorise. On repete le calcul NREPET fois pour mieux mesurer le temps d'execution
  auto start = std::chrono::high_resolution_clock::now();
  for (int repet = 0; repet < NREPET; repet++) {
    res = 0;
    for (int i = 0; i < dim; i++) res += tab0[i] * tab1[i];
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cout<<res<<'\n';
  std::chrono::duration<double> tempsSeq = end-start;
  std::cout << std::scientific << "Produit scalaire sans AVX: " << tempsSeq.count() / NREPET << "s" << std::endl;

  // Faire le produit scalaire vectorise AVX. On repete le calcul NREPET fois pour mieux mesurer le temps d'execution
  start = std::chrono::high_resolution_clock::now();
  for (int repet = 0; repet < NREPET; repet++) {
    __m256 res_tmp = _mm256_set1_ps(0.0f);
    res = 0;
    for (int i = 0; i < dim - 7; i += 8) {
      res_tmp = _mm256_add_ps(res_tmp, _mm256_mul_ps(_mm256_load_ps(tab0 + i), _mm256_load_ps(tab1 + i)));
    }
    float res_tab[8] __attribute__ ((aligned(32)));
    _mm256_store_ps(res_tab, res_tmp);
    for (int i = 0; i < 8; i++) res += res_tab[i];
  }
  end = std::chrono::high_resolution_clock::now();
  std::cout<<res<<'\n';
  std::chrono::duration<double> tempsAVX = end-start;
  std::cout << std::scientific << "Produit scalaire avec AVX: " << tempsAVX.count() / NREPET << "s" << std::endl;

  // Produit scalaire vectorise AVX FMA. On repete le calcul NREPET fois pour mieux mesurer le temps d'execution
  start = std::chrono::high_resolution_clock::now();
  

  for (int repet = 0; repet < NREPET; repet++) {
    res = 0;
    __m256 res_tmp = _mm256_set1_ps(0.0f);
    for (int i = 0; i < dim - 7; i += 8) {
      res_tmp = _mm256_fmadd_ps(_mm256_load_ps(tab0 + i), _mm256_load_ps(tab1 + i), res_tmp);
    }
    float res_tab[8] __attribute__ ((aligned(32)));
    _mm256_store_ps(res_tab, res_tmp);
    for (int i = 0; i < 8; i++) res += res_tab[i];
  }
  end = std::chrono::high_resolution_clock::now();
  std::cout<<res<<'\n';
  std::chrono::duration<double> tempsParAVXFMA = end-start;
  std::cout << std::scientific << "Produit scalaire avec AVX FMA: " << tempsParAVXFMA.count() / NREPET << "s" <<std::endl;

  // Produit scalaire vectorise AVX FMA et deroulement. On repete le calcul NREPET fois pour mieux mesurer le temps
  start = std::chrono::high_resolution_clock::now();
  for (int repet = 0; repet < NREPET; repet++) {
    __m256 res_tmp1 = _mm256_set1_ps(0.0f);
    __m256 res_tmp2 = _mm256_set1_ps(0.0f);
    __m256 res_tmp3 = _mm256_set1_ps(0.0f);
    __m256 res_tmp4 = _mm256_set1_ps(0.0f);
    res = 0;
    for (int i = 0; i < dim - 31; i+=32) {
      res_tmp1 = _mm256_fmadd_ps(_mm256_load_ps(tab0 + i), _mm256_load_ps(tab1 + i), res_tmp1);
      res_tmp2 = _mm256_fmadd_ps(_mm256_load_ps(tab0 + i + 8), _mm256_load_ps(tab1 + i + 8), res_tmp2);
      res_tmp3 = _mm256_fmadd_ps(_mm256_load_ps(tab0 + i + 16), _mm256_load_ps(tab1 + i + 16), res_tmp3);
      res_tmp4 = _mm256_fmadd_ps(_mm256_load_ps(tab0 + i + 24), _mm256_load_ps(tab1 + i + 24), res_tmp4);

    }
    float res_tab[8] __attribute__ ((aligned(32)));
    res_tmp1 = _mm256_add_ps(res_tmp1, res_tmp2);
    res_tmp3 = _mm256_add_ps(res_tmp3, res_tmp4);
    res_tmp1 = _mm256_add_ps(res_tmp1, res_tmp3);
    _mm256_store_ps(res_tab, res_tmp1);
    for (int i = 0; i < 8; i++) res += res_tab[i];
  }
  std::cout<<res<<std::endl;
  end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> tempsParAVXFMADeroule = end-start;
  std::cout << std::scientific << "Produit scalaire avec AVX FMA deroulement: " << tempsParAVXFMADeroule.count() /
    NREPET << "s" << std::endl;

  // Desallouer les tableaux tab0 et tab1
  _mm_free(tab0);
  _mm_free(tab1);

  return 0;
}
