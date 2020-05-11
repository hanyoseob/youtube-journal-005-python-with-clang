/*

nvcc -Xcompiler -fPIC math_cu.cu -shared -o libmath_cu.so

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <cuda.h>

#include "math_cu.cuh"

#define DLLEXPORT extern "C" __declspec(dllexport)


DLLEXPORT void mul_const(float *pddst, float *pdsrc, float dconst, int *pnsz) {
    float *gpddst = 0;
    float *gpdsrc = 0;
    int *gpnsz = 0;

    cudaMalloc((void **)&gpddst, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMalloc((void **)&gpdsrc, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMalloc((void **)&gpnsz, 3*sizeof(int));
    cudaMemset(gpddst, 0, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMemset(gpdsrc, 0, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMemset(gpnsz, 0, 3*sizeof(int));

    cudaMemcpy(gpdsrc, pdsrc, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpnsz, pnsz, 3*sizeof(int), cudaMemcpyHostToDevice);

    int nthread = 8;
    dim3 nblock(nthread, nthread, nthread);
    dim3 ngrid ((pnsz[0] + nthread - 1)/nthread,
               (pnsz[1] + nthread - 1)/nthread,
               (pnsz[2] + nthread - 1)/nthread);

    mul_const_kernel<<<ngrid, nblock>>>(gpddst, gpdsrc, dconst, gpnsz);

    cudaMemcpy(pddst, gpddst, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(gpddst);
    cudaFree(gpdsrc);
    cudaFree(gpnsz);

    gpddst = 0;
    gpdsrc = 0;
    gpnsz = 0;

    return ;
}

DLLEXPORT void add_const(float *pddst, float *pdsrc, float dconst, int *pnsz) {
    float *gpddst = 0;
    float *gpdsrc = 0;
    int *gpnsz = 0;

    cudaMalloc((void **)&gpddst, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMalloc((void **)&gpdsrc, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMalloc((void **)&gpnsz, 3*sizeof(int));
    cudaMemset(gpddst, 0, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMemset(gpdsrc, 0, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMemset(gpnsz, 0, 3*sizeof(int));

    cudaMemcpy(gpdsrc, pdsrc, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpnsz, pnsz, 3*sizeof(int), cudaMemcpyHostToDevice);

    int nthread = 8;
    dim3 nblock(nthread, nthread, nthread);
    dim3 ngrid ((pnsz[0] + nthread - 1)/nthread,
               (pnsz[1] + nthread - 1)/nthread,
               (pnsz[2] + nthread - 1)/nthread);

    add_const_kernel<<<ngrid, nblock>>>(gpddst, gpdsrc, dconst, gpnsz);

    cudaMemcpy(pddst, gpddst, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(gpddst);
    cudaFree(gpdsrc);
    cudaFree(gpnsz);

    gpddst = 0;
    gpdsrc = 0;
    gpnsz = 0;

    return ;
}

DLLEXPORT void mul_mat(float *pddst, float *pdsrc1, float *pdsrc2, int *pnsz) {
    float *gpddst = 0;
    float *gpdsrc1 = 0;
    float *gpdsrc2 = 0;
    int *gpnsz = 0;

    cudaMalloc((void **)&gpddst, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMalloc((void **)&gpdsrc1, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMalloc((void **)&gpdsrc2, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMalloc((void **)&gpnsz, 3*sizeof(int));
    cudaMemset(gpddst, 0, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMemset(gpdsrc1, 0, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMemset(gpdsrc2, 0, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMemset(gpnsz, 0, 3*sizeof(int));

    cudaMemcpy(gpdsrc1, pdsrc1, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpdsrc2, pdsrc2, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpnsz, pnsz, 3*sizeof(int), cudaMemcpyHostToDevice);

    int nthread = 8;
    dim3 nblock(nthread, nthread, nthread);
    dim3 ngrid ((pnsz[0] + nthread - 1)/nthread,
               (pnsz[1] + nthread - 1)/nthread,
               (pnsz[2] + nthread - 1)/nthread);

    mul_mat_kernel<<<ngrid, nblock>>>(gpddst, gpdsrc1, gpdsrc2, gpnsz);

    cudaMemcpy(pddst, gpddst, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(gpddst);
    cudaFree(gpdsrc1);
    cudaFree(gpdsrc2);
    cudaFree(gpnsz);

    gpddst = 0;
    gpdsrc1 = 0;
    gpdsrc2 = 0;
    gpnsz = 0;

    return ;
}

DLLEXPORT void add_mat(float *pddst, float *pdsrc1, float *pdsrc2, int *pnsz) {
    float *gpddst = 0;
    float *gpdsrc1 = 0;
    float *gpdsrc2 = 0;
    int *gpnsz = 0;

    cudaMalloc((void **)&gpddst, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMalloc((void **)&gpdsrc1, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMalloc((void **)&gpdsrc2, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMalloc((void **)&gpnsz, 3*sizeof(int));
    cudaMemset(gpddst, 0, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMemset(gpdsrc1, 0, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMemset(gpdsrc2, 0, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMemset(gpnsz, 0, 3*sizeof(int));

    cudaMemcpy(gpdsrc1, pdsrc1, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpdsrc2, pdsrc2, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpnsz, pnsz, 3*sizeof(int), cudaMemcpyHostToDevice);

    int nthread = 8;
    dim3 nblock(nthread, nthread, nthread);
    dim3 ngrid ((pnsz[0] + nthread - 1)/nthread,
               (pnsz[1] + nthread - 1)/nthread,
               (pnsz[2] + nthread - 1)/nthread);

    add_mat_kernel<<<ngrid, nblock>>>(gpddst, gpdsrc1, gpdsrc2, gpnsz);

    cudaMemcpy(pddst, gpddst, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(gpddst);
    cudaFree(gpdsrc1);
    cudaFree(gpdsrc2);
    cudaFree(gpnsz);

    gpddst = 0;
    gpdsrc1 = 0;
    gpdsrc2 = 0;
    gpnsz = 0;

    return ;
}

















