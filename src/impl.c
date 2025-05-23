#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <omp.h>
int n;
inline int d(int i, int j){
    return i * n + j;
}
inline int rol(int x, int y){
    return (x + y + 1) / 2;
}
inline int col(int x, int y){
    return (x - y + n - 1) / 2;
}
inline int f(int x, int y){
    return d(rol(x, y), col(x, y));
}
inline int max(int a, int b){
    return a > b ? a : b;
}
inline int min(int a, int b){
    return a < b ? a : b;
}
void impl(int N, int step, double *p) {
    n = N;
    double *p1 = calloc((N + 1) * N, sizeof(double));
    double *p2 = calloc((N + 1) * N, sizeof(double));
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            if((i + j) % 2 == 0) p1[f(i, j)] = p[d(i, j)];
            else p2[f(i, j)] = p[d(i, j)];
        }
    }
    for(int k = 1; k <= step; k++){
        if(k % 2){
        #pragma omp parallel for schedule(guided, (N - 2) / (4 * omp_get_num_threads()))
        for(int s = 2; s <= 2 * N - 4; s += 2){
            for(int j = max(1, s - N + 2); j <= min(s - 1, N - 2); j++){
                int i = s - j;
                p1[f(i, j)] = (p2[f(i - 1, j)] + p2[f(i + 1, j)] + p2[f(i, j - 1)] + p2[f(i, j + 1)]) / 4.0f;
            }
        }
        }else{
        #pragma omp parallel for schedule(guided, (N - 2) / (4 * omp_get_num_threads()))
        for(int s = 3; s <= 2 * N - 4; s += 2){
            for(int j = max(1, s - N + 2); j <= min(s - 1, N - 2); j++){
                int i = s - j;
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