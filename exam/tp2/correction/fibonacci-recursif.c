#include <stdio.h>
#include <stdlib.h>
#include "omp.h"

int fibo(int n) {
  if (n < 2) { return n; }
  int x, y;
#pragma omp task shared(x)
  x = fibo(n - 2);

  y = fibo(n - 1);
#pragma omp taskwait
  return x + y;
}

int main(int argc, char **argv)
{
  int n = atoi(argv[1]);
#pragma omp parallel
  {
#pragma omp single
    printf("fibo(%d) = %d\n", n, fibo(n));
  }
  return 0;
}
