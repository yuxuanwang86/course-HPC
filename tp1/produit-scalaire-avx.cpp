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

inline float s_sp(float* A, float* B, const int dim) {
  float res = 0.0f;
  for (size_t i = 0; i < dim; i++) res += A[i] * B[i];
  return res;
}

void v_sp(float *A, float *B, const int dim) {
  __m256 res = _mm256_set1_ps(0.f);
  for (int i = 0; i < dim; i += 8) {
    __m256 buf_A = _mm256_load_ps(A+i);
    __m256 buf_B = _mm256_load_ps(B+i);
    __m256 buf = _mm256_mul_ps(buf_A, buf_B);
    res = _mm256_add_ps(res, buf);
  }
  float res_tab[8] __attribute__((aligned(32))); 
  _mm256_store_ps(res_tab, res);
  float re = 0.f;
  for (int i = 0; i < 8; i++) re += res_tab[i];
}

void fv_sp(float *A, float *B, const int dim) {
  __m256 res1 = _mm256_set1_ps(0.f);
  __m256 res2 = _mm256_set1_ps(0.f);
  __m256 res3 = _mm256_set1_ps(0.f);
  __m256 res4 = _mm256_set1_ps(0.f);
  for (int i = 0; i < dim; i += 32) {
    __m256 buf_A1 = _mm256_load_ps(A+i);
    __m256 buf_A2 = _mm256_load_ps(A+i+8);
    __m256 buf_A3 = _mm256_load_ps(A+i+16);
    __m256 buf_A4 = _mm256_load_ps(A+i+24);
    __m256 buf_B1 = _mm256_load_ps(B+i);
    __m256 buf_B2 = _mm256_load_ps(B+i+8);
    __m256 buf_B3 = _mm256_load_ps(B+i+16);
    __m256 buf_B4 = _mm256_load_ps(B+i+24);
    __m256 buf1 = _mm256_mul_ps(buf_A1, buf_B1);
    __m256 buf2 = _mm256_mul_ps(buf_A2, buf_B2);
    __m256 buf3 = _mm256_mul_ps(buf_A3, buf_B3);
    __m256 buf4 = _mm256_mul_ps(buf_A4, buf_B4);
    res1 = _mm256_add_ps(res1, buf1);
    res2 = _mm256_add_ps(res2, buf2);
    res3 = _mm256_add_ps(res3, buf3);
    res4 = _mm256_add_ps(res4, buf4);
  }
  res1 = _mm256_add_ps(res1, res2);
  res3 = _mm256_add_ps(res3, res4);
  res1 = _mm256_add_ps(res1, res3);
  float res_tab[8] __attribute__((aligned(32))); 
  _mm256_store_ps(res_tab, res1);
  float re = 0.f;
  for (int i = 0; i < 8; i++) re += res_tab[i];
}

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
  // for (int i = 0; i < 20; i++) std::cout<<tab0[i]<<' ';
  // std::cout<<'\n';
  // Q2
  // Faire le produit scalaire non-vectorise. On repete le calcul NREPET fois pour mieux mesurer le temps d'execution
  auto start = std::chrono::high_resolution_clock::now();
  float res = 0.0f;
  for (int repet = 0; repet < NREPET; repet++) {
    float res = 0.0f;
    for (size_t i = 0; i < dim; i++) res +=  tab0[i] + tab1[i];
  }
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
  }
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
  }
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
  }
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
