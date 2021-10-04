#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <assert.h>
#include <iostream>
#include <chrono>

#define SWAP(a,b) {int tmp = a; a = b; b = tmp;}
#define SIZE 1000000000


void verify(int* a, int size);
void merge(int* a, int size, int* temp);
void mergesort(int* a, int size, int* temp, const int& K);

int main(int argc, char** argv) {
  const int K = std::atoi(argv[1]);
  int* a;
  int* temp;
  double inicio, fim;
  int size = SIZE;
  int i;

  //setup
  if (argc>=2) size = atoi(argv[1]);
  printf("Sorting an array of size %d.\n", size);
  a = (int*)calloc(size, sizeof(int));
  temp = (int*)calloc(size, sizeof(int));
  srand(26);
  for (i = 0; i<size; ++i) {
    a[i] = rand();
  }

  //sort
  auto start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel
  #pragma omp single
  mergesort(a, size, temp, K);
  std::chrono::duration<double> temps = std::chrono::high_resolution_clock::now() - start;
  std::cout << std::scientific << "Temps d'execution: " << temps.count() << "s" << std::endl;
  verify(a, size);
}

void verify(int* a, int size) {
  int sorted = 1;
  int i;

  for (i = 0; i < size-1; ++i) sorted &= (a[i] <= a[i+1]);

  if (sorted) printf("The array was properly sorted.\n");
  else printf("There was an error when sorting the array.\n");
}

void merge(int* a, int size, int* temp) {
  int i1 = 0;
  int i2 = size / 2;
  int it = 0;

  while(i1 < size / 2 && i2 < size) {
    if (a[i1] <= a[i2]) {
      temp[it] = a[i1];
      i1 += 1;
    }
    else {
      temp[it] = a[i2];
      i2 += 1;
    }
    it += 1;
  }

  while (i1 < size/2) {
    temp[it] = a[i1];
    i1++;
    it++;
  }
  while (i2 < size) {
    temp[it] = a[i2];
    i2++;
    it++;
  }

  memcpy(a, temp, size*sizeof(int));
}

void mergesort(int* a, int size, int* temp, const int& K) {
  if (size < 2) return;   //nothing to sort
  if (size == 2) {        //only two values to sort
    if (a[0] <= a[1])
      return;
    else {
      SWAP(a[0], a[1]);
      return;
    }
  } else {                //mergesort
    #pragma omp task if (size > K)
    mergesort(a, size/2, temp, K);
    #pragma omp task if (size > K)
    mergesort(a + size/2, size - size/2, temp + size/2, K); //a + size/2: pointer arithmetic
    #pragma omp taskwait
    merge(a, size, temp);
  }
  return;
}
