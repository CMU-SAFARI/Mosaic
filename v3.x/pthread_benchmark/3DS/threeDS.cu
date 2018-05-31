#include <stdio.h>
#include "../benchmark_common.h"
#include "reference.h"

//__constant__ float c_coeff[10];

#define BLOCK_DIMX 32
#define BLOCK_DIMY 8
#define RADIUS 8  // 4
#define SMEM_DIMX (BLOCK_DIMX + 2 * RADIUS)

extern "C" __global__ void stencil_3D_order8(float* g_output,
                                             float* g_input,
                                             float* g_coeff,
                                             const int dimx,
                                             const int dimy,
                                             const int dimz,
                                             float flag) {
  __shared__ float s_data[BLOCK_DIMY + 2 * RADIUS][BLOCK_DIMX + 2 * RADIUS];

  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int in_idx = iy * dimx + ix;
  int out_idx = 0;
  int stride = dimx * dimy;

  float infront1, infront2, infront3, infront4;
  float behind1, behind2, behind3, behind4;
  float current;

  int tx = threadIdx.x + RADIUS;
  int ty = threadIdx.y + RADIUS;

  // fill the "in-front" and "behind" data
  behind3 = g_input[in_idx];
  in_idx += stride;
  behind2 = g_input[in_idx];
  in_idx += stride;
  behind1 = g_input[in_idx];
  in_idx += stride;

  current = g_input[in_idx];
  out_idx = in_idx;
  in_idx += stride;

  infront1 = g_input[in_idx];
  in_idx += stride;
  infront2 = g_input[in_idx];
  in_idx += stride;
  infront3 = g_input[in_idx];
  in_idx += stride;
  infront4 = g_input[in_idx];
  in_idx += stride;

  for (int i = RADIUS; i < dimz - RADIUS; i++) {
    //////////////////////////////////////////
    // advance the slice (move the thread-front)
    behind4 = behind3;
    behind3 = behind2;
    behind2 = behind1;
    behind1 = current;
    current = infront1;
    infront1 = infront2;
    infront2 = infront3;
    infront3 = infront4;
    infront4 = g_input[in_idx];

    in_idx += stride;
    out_idx += stride;
    __syncthreads();

    /////////////////////////////////////////
    // update the data slice in smem

    if (threadIdx.y < RADIUS)  // halo above/below
    {
      s_data[threadIdx.y][tx] = g_input[out_idx - RADIUS * dimx];
      s_data[threadIdx.y + BLOCK_DIMY + RADIUS][tx] =
          g_input[out_idx + BLOCK_DIMY * dimx];
    }

    if (threadIdx.x < RADIUS)  // halo left/right
    {
      s_data[ty][threadIdx.x] = g_input[out_idx - RADIUS];
      s_data[ty][threadIdx.x + BLOCK_DIMX + RADIUS] =
          g_input[out_idx + BLOCK_DIMX];
    }

    // update the slice in smem
    s_data[ty][tx] = current;
    __syncthreads();

    /////////////////////////////////////////
    // compute the output value
    float value = g_coeff[0] * current;
    value += g_coeff[1] *
             (infront1 + behind1 + s_data[ty - 1][tx] + s_data[ty + 1][tx] +
              s_data[ty][tx - 1] + s_data[ty][tx + 1]);
    value += g_coeff[2] *
             (infront2 + behind2 + s_data[ty - 2][tx] + s_data[ty + 2][tx] +
              s_data[ty][tx - 2] + s_data[ty][tx + 2]);
    value += g_coeff[3] *
             (infront3 + behind3 + s_data[ty - 3][tx] + s_data[ty + 3][tx] +
              s_data[ty][tx - 3] + s_data[ty][tx + 3]);
    value += g_coeff[4] *
             (infront4 + behind4 + s_data[ty - 4][tx] + s_data[ty + 4][tx] +
              s_data[ty][tx - 4] + s_data[ty][tx + 4]);
    g_output[out_idx] = value;
  }
}

void timing_experiment(void (*kernel)(float*,
                                      float*,
                                      float*,
                                      const int,
                                      const int,
                                      const int,
                                      float),
                       float* d_output,
                       float* d_input,
                       float* d_coeff,
                       int dimx,
                       int dimy,
                       int dimz,
                       cudaStream_t stream_app,
                       pthread_mutex_t* mutexapp,
                       bool flag,
                       int smem = 0,
                       int nreps = 1,
                       float mult = 1.f) {
  dim3 block(BLOCK_DIMX, BLOCK_DIMY);
  dim3 grid(dimx / block.x, dimy / block.y);
  printf("(%d,%d)x(%d,%d) grid\n", grid.x, grid.y, block.x, block.y);

  for (int i = 0; i < nreps; i++) {
    kernel<<<grid, block, smem, stream_app>>>(d_output, d_input, d_coeff, dimx,
                                              dimy, dimz, 0.f);
    pthread_mutex_unlock(mutexapp);
    if (flag)
      cutilSafeCall(cudaStreamSynchronize(stream_app));
    pthread_mutex_lock(mutexapp);
  }
  pthread_mutex_unlock(mutexapp);
}

int pad;
int dimx;
int dimy;
int dimz;
int nreps;
int nbytes;

float *d_input = 0, *d_output = 0;
float *h_data = 0, *h_reference = 0;

float h_coeff_symmetric[10] = {1.f, 1.f, 1.f, 1.f, 1.f,
                               1.f, 1.f, 1.f, 1.f, 1.f};
float* d_coeff;

int main_threeDS(cudaStream_t stream_app,
                 pthread_mutex_t* mutexapp,
                 bool flag) {
  pad = 0;
  dimx = 128 * 2;  // 640+pad;
  dimy = 64 * 2;   // 480;
  dimz = 64 * 2;   // 100;
  nreps = 1;
  nbytes = dimx * dimy * dimz * sizeof(float);
  cudaMalloc((void**)&d_input, nbytes);
  cudaMalloc((void**)&d_output, nbytes);
  printf("CUDA: %s\n", cudaGetErrorString(cudaGetLastError()));
  if (0 == d_input || 0 == d_output) {
    printf("couldn't allocate all GPU memory: %2.f MB\n",
           (2.f * nbytes) / (1024.f * 1024.f));
    exit(1);
  }
  printf("allocated %.1f MB on device\n", (2.f * nbytes) / (1024.f * 1024.f));
  h_data = (float*)malloc(nbytes);
  h_reference = (float*)malloc(nbytes);
  if (0 == h_data || 0 == h_reference) {
    printf("couldn't allocate CPU memory\n");
    exit(1);
  }
  random_data(h_data, dimx, dimy, dimz, 1, 5);
  cudaMemcpyAsync(d_input, h_data, nbytes, cudaMemcpyHostToDevice, stream_app);
  cudaMemcpyAsync(d_output, h_data, nbytes, cudaMemcpyHostToDevice, stream_app);

  // setup coefficients
  cudaMalloc((void**)&d_coeff, 10 * sizeof(float));
  cudaMemcpyAsync(d_coeff, h_coeff_symmetric, 10 * sizeof(float),
                  cudaMemcpyHostToDevice, stream_app);

  dim3 block(BLOCK_DIMX, BLOCK_DIMY);
  dim3 grid(dimx / block.x, dimy / block.y);
  printf("(%d,%d)x(%d,%d) grid\n", grid.x, grid.y, block.x, block.y);

  printf("%20s ", "FD_full_2D");
  timing_experiment(stencil_3D_order8, d_output, d_input, d_coeff, dimx, dimy,
                    dimz, stream_app, mutexapp, flag, 0, nreps,
                    (3.f * dimz - 4 * RADIUS));

  if (d_input)
    cudaFree(d_input);
  if (d_output)
    cudaFree(d_output);

  return 0;
}
