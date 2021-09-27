/**
  * Produit scalaire de deux tableaux avec les intrinseques AVX et FMA.
  * A compiler avec les drapeaux -O2 -mavx2 -mfma
  */

#include <immintrin.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>

#define NREPET 4096

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

  // Q1
  // Allouer les tableaux tab0 et tab1 de flottants de taille dim alignes par 32 octets, puis initialiser tab0[i]=i
  float* tab0 = (float*) _mm_malloc (dim * sizeof(float), 32);
  float* tab1 = (float*) _mm_malloc (dim * sizeof(float), 32);
  for (int i = 0; i < dim; i++) {
    tab0[i] = i;
    tab1[i] = i;
  }
  //for (int i = 0; i < 20; i++) std::cout<<tab0[i]<<' ';
  ///std::cout<<'\n';
  // Q2
  // Faire le produit scalaire non-vectorise. On repete le calcul NREPET fois pour mieux mesurer le temps d'execution
  auto start = std::chrono::high_resolution_clock::now();
  float final_res = 0.0f;
  for (int repet = 0; repet < NREPET; repet++) {
    final_res = 0.0f;
    for (size_t i = 0; i < dim; i++) final_res +=  tab0[i] + tab1[i];
  }
  std::cout << final_res << '\n';
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> tempsSeq = end-start;
  std::cout << std::scientific << "Produit scalaire sans AVX: " << tempsSeq.count() / NREPET << "s" << std::endl;

  // Q3
  // Faire le produit scalaire vectorise AVX. On repete le calcul NREPET fois pour mieux mesurer le temps d'execution
  start = std::chrono::high_resolution_clock::now();
  for (int repet = 0; repet < NREPET; repet++) {
    
    float res[8] __attribute__((aligned(32)));
    __m256 res_buf = _mm256_set1_ps(0.0f);
    for (size_t i = 0; i < dim; i += 8) {
      res_buf = _mm256_add_ps(res_buf, _mm256_mul_ps(_mm256_load_ps(tab0 + i), _mm256_load_ps(tab1 + i)));
    }
    _mm256_store_ps(res, res_buf);
    final_res = 0.0f;
    for (size_t j = 0; j < 8; ++j) final_res += res[j];
  }
  std::cout << final_res << '\n';
  end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> tempsAVX = end-start;
  std::cout << std::scientific << "Produit scalaire avec AVX: " << tempsAVX.count() / NREPET << "s" << std::endl;

  // Q4
  // Produit scalaire vectorise AVX FMA. On repete le calcul NREPET fois pour mieux mesurer le temps d'execution
  start = std::chrono::high_resolution_clock::now();
  for (int repet = 0; repet < NREPET; repet++) {
    float res[8] __attribute__((aligned(32)));
    __m256 res_buf = _mm256_set1_ps(0.0f);
    for (size_t i = 0; i < dim; i += 8) {
      res_buf = _mm256_fmadd_ps(_mm256_load_ps(tab0 + i), _mm256_load_ps(tab1 + i), res_buf);
    }
    _mm256_store_ps(res, res_buf);
    final_res = 0.0f;
    for (size_t j = 0; j < 8; ++j) final_res += res[j];
  }
  std::cout << final_res << '\n';
  end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> tempsParAVXFMA = end-start;
  std::cout << std::scientific << "Produit scalaire avec AVX FMA: " << tempsParAVXFMA.count() / NREPET << "s" <<
    std::endl;

  // Produit scalaire vectorise AVX FMA et deroulement. On repete le calcul NREPET fois pour mieux mesurer le temps
  start = std::chrono::high_resolution_clock::now();
  for (int repet = 0; repet < NREPET; repet++) {
    __m256 res_buf1 = _mm256_set1_ps(0.0f);
    __m256 res_buf2 = _mm256_set1_ps(0.0f);
    __m256 res_buf3 = _mm256_set1_ps(0.0f);
    __m256 res_buf4 = _mm256_set1_ps(0.0f);
    for (size_t i = 0; i < dim; i += 32) {
      res_buf1 = _mm256_fmadd_ps(_mm256_load_ps(tab0 + i), _mm256_load_ps(tab1 + i), res_buf1);
      res_buf2 = _mm256_fmadd_ps(_mm256_load_ps(tab0 + i + 8), _mm256_load_ps(tab1 + i + 8), res_buf2);
      res_buf3 = _mm256_fmadd_ps(_mm256_load_ps(tab0 + i + 16), _mm256_load_ps(tab1 + i + 16), res_buf3);
      res_buf4 = _mm256_fmadd_ps(_mm256_load_ps(tab0 + i + 24), _mm256_load_ps(tab1 + i + 24), res_buf4);
    }
    res_buf1 = _mm256_add_ps(res_buf1, res_buf2);
    res_buf3 = _mm256_add_ps(res_buf3, res_buf4);
    res_buf1 = _mm256_add_ps(res_buf1, res_buf3);
    float res[8] __attribute__((aligned(32)));
    _mm256_store_ps(res, res_buf1);
    final_res = 0.0f;
    for (size_t j = 0; j < 8; ++j) final_res += res[j];
  }
  std::cout << final_res << '\n';
  end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> tempsParAVXFMADeroule = end-start;
  std::cout << std::scientific << "Produit scalaire avec AVX FMA deroulement: " << tempsParAVXFMADeroule.count() /
    NREPET << "s" << std::endl;

  // Desallouer les tableaux tab0 et tab1
  // A FAIRE ...
  _mm_free(tab1);
  _mm_free(tab0);
  return 0;
}
