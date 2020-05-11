
extern "C" __global__ void mul_const_kernel(float *pddst, float *pdsrc, float dconst, int *pnsz) {
    int nx = pnsz[0];
    int ny = pnsz[1];
    int nz = pnsz[2];
    int id = 0;

    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    int idy = blockDim.y*blockIdx.y + threadIdx.y;
    int idz = blockDim.z*blockIdx.z + threadIdx.z;

    if (idx >= nx || idy >= ny || idz >= nz) return ;

    id = ny*nx*idz + nx*idy + idx;
    pddst[id] = pdsrc[id] * dconst;

    return ;
}

extern "C" __global__ void add_const_kernel(float *pddst, float *pdsrc, float dconst, int *pnsz) {
    int nx = pnsz[0];
    int ny = pnsz[1];
    int nz = pnsz[2];
    int id = 0;

    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    int idy = blockDim.y*blockIdx.y + threadIdx.y;
    int idz = blockDim.z*blockIdx.z + threadIdx.z;

    if (idx >= nx || idy >= ny || idz >= nz) return ;

    id = ny*nx*idz + nx*idy + idx;
    pddst[id] = pdsrc[id] + dconst;

    return ;
}

extern "C" __global__ void mul_mat_kernel(float *pddst, float *pdsrc1, float *pdsrc2, int *pnsz) {
    int nx = pnsz[0];
    int ny = pnsz[1];
    int nz = pnsz[2];
    int id = 0;

    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    int idy = blockDim.y*blockIdx.y + threadIdx.y;
    int idz = blockDim.z*blockIdx.z + threadIdx.z;

    if (idx >= nx || idy >= ny || idz >= nz) return ;

    id = ny*nx*idz + nx*idy + idx;
    pddst[id] = pdsrc1[id] * pdsrc2[id];

    return ;
}

extern "C" __global__ void add_mat_kernel(float *pddst, float *pdsrc1, float *pdsrc2, int *pnsz) {
    int nx = pnsz[0];
    int ny = pnsz[1];
    int nz = pnsz[2];
    int id = 0;

    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    int idy = blockDim.y*blockIdx.y + threadIdx.y;
    int idz = blockDim.z*blockIdx.z + threadIdx.z;

    if (idx >= nx || idy >= ny || idz >= nz) return ;

    id = ny*nx*idz + nx*idy + idx;
    pddst[id] = pdsrc1[id] + pdsrc2[id];

    return ;
}