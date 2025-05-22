#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#define d(i, j) ((i) * N + (j))
#define f(i, j) (((i + 1) / 2) * N + (j))
void impl(int N, int step, double *p) {
  double *p1 = calloc((N / 2 + 2) * N, sizeof(double));
  double *p2 = calloc((N / 2 + 2) * N, sizeof(double));
  for(int i = 0; i < N; i++){
    for(int j = 0; j < N; j++){
      if((i + j) % 2 == 0) p1[f(i, j)] = p[d(i, j)];
      else p2[f(i, j)] = p[d(i, j)];
    }
  }
  for(int k = 1; k <= step; k++){
    if(k % 2){
      #pragma omp parallel for
      for(int i = 1; i < N - 1; i++){
        for(int j = 2 - i % 2; j < N - 1; j += 2){
          p1[f(i, j)] = (p2[f(i - 1, j)] + p2[f(i + 1, j)] + p2[f(i, j - 1)] + p2[f(i, j + 1)]) / 4.0f;
        }
      }
    }else{
      #pragma omp parallel for
      for(int i = 1; i < N - 1; i++){
        for(int j = i % 2 + 1; j < N - 1; j += 2){
          p2[f(i, j)] = (p1[f(i - 1, j)] + p1[f(i + 1, j)] + p1[f(i, j - 1)] + p1[f(i, j + 1)]) / 4.0f;
        }
      }
    }
  }
  for(int i = 0; i < N; i++){
    for(int j = 0; j < N; j++){
      if((i + j) % 2 == 0) p[d(i, j)] = p1[f(i, j)];
      else p[d(i, j)] = p2[f(i, j)];
    }
  }
  free(p1);
  free(p2);
}