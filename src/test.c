#include <immintrin.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int N = 2502;    // length of the square region
int step = 2000; // Number of update steps
#define EPS 1e-5

void baseline(int N, int step, double *p, double *p_next);
#ifdef NOCUDA
void impl_row(int N, int step, double *p);
void impl_col(int N, int step, double *p);
void impl_col_SIMD(int N, int step, double *p);
#else
void impl_CUDA(int N, int step, double *p);
#endif

double record [100];
int rp = 0;
void display_time(struct timespec start, struct timespec end) {
  record[rp++] = (double)(end.tv_sec - start.tv_sec) * 1000.0f +
                       (double)(end.tv_nsec - start.tv_nsec) / 1000000.0f;
  printf("%fms\n", record[rp - 1]);
}

bool is_legal_answer(int N, double *ref_p, double *ref_p_next, double *p) {
  double diff = 0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (fabs(ref_p[i * N + j] - p[i * N + j]) <
          fabs(ref_p_next[i * N + j] - p[i * N + j])) {
        diff += (ref_p[i * N + j] - p[i * N + j]) *
                (ref_p[i * N + j] - p[i * N + j]);
      } else {
        diff += (ref_p_next[i * N + j] - p[i * N + j]) *
                (ref_p_next[i * N + j] - p[i * N + j]);
      }
      if (diff > EPS) {
        return false;
      }
    }
  }
  return true;
}

void test(bool baseline_flag) {
  double *ref_p = calloc(N * N, sizeof(double));
  double *ref_p_next = calloc(N * N, sizeof(double));
  double *p_rol = calloc(N * N, sizeof(double));
  double *p = calloc(N * N, sizeof(double));

  // initial values for grid
  for (int i = 0; i < N; i++) {
    ref_p[i] = 1.0f;
  }
  for (int j = 0; j < N; j++) {
    ref_p[j * N] = 1.0f;
  }

  memcpy(ref_p_next, ref_p, N * N * sizeof(double));
  memcpy(p, ref_p, N * N * sizeof(double));
  memcpy(p_rol, ref_p, N * N * sizeof(double));

  struct timespec start, end;

  // baseline
  clock_gettime(CLOCK_MONOTONIC, &start);
  if(baseline_flag)baseline(N, step, ref_p, ref_p_next);
  clock_gettime(CLOCK_MONOTONIC, &end);
  printf("Baseline: ");
  display_time(start, end);

  clock_gettime(CLOCK_MONOTONIC, &start);
#ifdef NOCUDA
  impl_col_SIMD(N, step, p_rol);
#else
  impl_CUDA(N, step, p_rol);
#endif
  clock_gettime(CLOCK_MONOTONIC, &end);
  printf("Impl_row: ");
  display_time(start, end);

  // your implementation
  clock_gettime(CLOCK_MONOTONIC, &start);
  // impl(N, step, p);
  clock_gettime(CLOCK_MONOTONIC, &end);
  printf("Yours:    ");
  display_time(start, end);
  if (is_legal_answer(N, ref_p, ref_p_next, p_rol)) {
    puts("-------Pass-------");
  } else {
    puts("x-x-x-Invalid-x-x-x");
  }
  free(ref_p);
  free(ref_p_next);
  free(p_rol);
  free(p);
}

int main(void) {
  double s1 = 0, s2 = 0;
  for(int i=1;i<=10;i++){
    test(0);
    s1 += record[1];
    s2 += record[2];
    rp = 0;
  }
  printf("Impl_rol average: ");
  printf("%fms\n", s1/10.0f);
  printf("Yours average: ");
  printf("%fms\n", s2/10.0f);
  return 0;
}
