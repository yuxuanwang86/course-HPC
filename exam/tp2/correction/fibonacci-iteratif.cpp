#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "omp.h"

int main(int argc, char **argv)
{
  int n = atoi(argv[1]);
  int fib[1000];
  fib[0] = 0;
  fib[1] = 1;
#pragma omp parallel
  {
#pragma omp single
    for (int i = 2; i < n; i++) {
#pragma omp task depend(in:fib[i - 1], fib[i - 2]) depend(out:fib[i])
      {
        fib[i] = fib[i - 1] + fib[i - 2];
      }
    }
  }
  for (int i = 0; i < n; i++) { printf("%d ", fib[i]); }
  printf("\n");
  return 0;
}
