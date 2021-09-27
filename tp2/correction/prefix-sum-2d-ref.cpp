#include <cstdio>
#include <iostream>
#include "omp.h"

#define N 16
#define B 4
#define NTACHES (N / B)

double A[N][N];
bool deps[NTACHES + 1][NTACHES + 1];

void afficherTableau()
{
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      printf("%4.0lf ", A[i][j]);
    }
    printf("\n");
  }
  printf("\n");
}

int main(int argc, char **argv)
{
  // Initialiser le tableau A[i][j] = i + j en parallele avec des tuiles (B x B) de taches
  int numBlocs = N / B;
#pragma omp parallel
  {
#pragma omp for collapse(2)
    for (int ti = 0; ti < numBlocs; ti++) {
      for (int tj = 0; tj < numBlocs; tj++) {
#pragma omp task firstprivate(ti, tj) shared(numBlocs)
        for (int i = ti * B; i < (ti + 1) * B; i++) {
          for (int j = tj * B; j < (tj + 1) * B; j++) {
            A[i][j] = i + j;
          }
        }
      }
    }
#pragma omp single
  afficherTableau();
#pragma omp single
    for (int ti = 0; ti < numBlocs; ti++) {
      for (int tj = 0; tj < numBlocs; tj++) {
#pragma omp task firstprivate(ti, tj) shared(numBlocs) depend(out:deps[ti + 1][tj + 1]) \
        depend(in:deps[ti][tj],deps[ti + 1][tj], deps[ti][tj + 1])
        for (int i = ti * B; i < (ti + 1) * B; i++) {
          for (int j = tj * B; j < (tj + 1) * B; j++) {
            A[i][j] += (i == 0) ? A[i][j - 1] :
              (j == 0) ? A[i - 1][j] :
              (A[i - 1][j] + A[i][j - 1] - A[i - 1][j - 1]);
          }
        }
      }
    }
  }
  afficherTableau();
  return 0;
}
