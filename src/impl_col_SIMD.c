#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>
#define d(i, j) ((i) * N + (j))
#define f(i, j) ((i) * N / 2 + ((j) / 2))
void impl_col_SIMD(int N, int step, double *p) {
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
            #pragma omp parallel for //schedule(dynamic, (N - 2) / ( 4 * omp_get_num_threads()))
            for(int i = 1; i < N - 1; i++){
                int j = 2 - i % 2;
                for(;j + 6 < N - 1; j += 8){
                    __m256d s1 = _mm256_loadu_pd(&p2[f(i - 1, j)]);
                    __m256d s2 = _mm256_loadu_pd(&p2[f(i + 1, j)]);
                    __m256d s3 = _mm256_loadu_pd(&p2[f(i, j - 1)]);
                    __m256d s4 = _mm256_loadu_pd(&p2[f(i, j + 1)]);
                    __m256d r1 = _mm256_add_pd(s1, s2);
                    __m256d r2 = _mm256_add_pd(s3, s4);
                    __m256d res = _mm256_add_pd(r1, r2);
                    res = _mm256_mul_pd(res, _mm256_set1_pd(0.25f));
                    _mm256_storeu_pd(&p1[f(i, j)], res);
                }
                for(;j < N - 1; j += 2){
                    p1[f(i, j)] = (p2[f(i - 1, j)] + p2[f(i + 1, j)] + p2[f(i, j - 1)] + p2[f(i, j + 1)]) / 4.0f;
                }
            }
        }else{
            #pragma omp parallel for //schedule(dynamic, (N - 2) / ( 4 * omp_get_num_threads()))
            for(int i = 1; i < N - 1; i++){
                int j = i % 2 + 1;
                for(;j + 6 < N - 1; j += 8){
                    __m256d s1 = _mm256_loadu_pd(&p1[f(i - 1, j)]);
                    __m256d s2 = _mm256_loadu_pd(&p1[f(i + 1, j)]);
                    __m256d s3 = _mm256_loadu_pd(&p1[f(i, j - 1)]);
                    __m256d s4 = _mm256_loadu_pd(&p1[f(i, j + 1)]);
                    __m256d r1 = _mm256_add_pd(s1, s2);
                    __m256d r2 = _mm256_add_pd(s3, s4);
                    __m256d res = _mm256_add_pd(r1, r2);
                    res = _mm256_mul_pd(res, _mm256_set1_pd(0.25f));
                    _mm256_storeu_pd(&p2[f(i, j)], res);
                }
                for(;j < N - 1; j += 2){
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