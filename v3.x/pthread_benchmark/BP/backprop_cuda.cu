

// includes, system
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

// includes, kernels
#include "../benchmark_common.h"
#include "backprop.h"
#include "backprop_cuda_kernel.cu"

////////////////////////////////////////////////////////////////////////////////

double gettime_bp() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + t.tv_usec * 1e-6;
}

unsigned int num_threads_b = 0;
unsigned int num_blocks_b = 0;

extern "C" void bpnn_train_cuda(BPNN* net,
                                float* eo,
                                float* eh,
                                cudaStream_t stream_app,
                                pthread_mutex_t* mutexapp,
                                bool flag) {
  int in, hid, out;
  float out_err, hid_err;

  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;

#ifdef GPU
  int m = 0;
  float* input_hidden_cuda;
  float* input_cuda;
  float* output_hidden_cuda;
  float* partial_sum;
  float* hidden_partial_sum;
  float* hidden_delta_cuda;
  float* input_prev_weights_cuda;
  float sum;
  float* input_weights_one_dim;
  float* input_weights_prev_one_dim;
  num_blocks_b = in / 16;
  dim3 grid(1, num_blocks_b);
  dim3 threads(16, 16);

  input_weights_one_dim = (float*)malloc((in + 1) * (hid + 1) * sizeof(float));
  input_weights_prev_one_dim =
      (float*)malloc((in + 1) * (hid + 1) * sizeof(float));
  partial_sum = (float*)malloc(num_blocks_b * WIDTH * sizeof(float));

  // this preprocessing stage is added to correct the bugs of wrong memcopy
  // using two-dimensional net->inputweights
  for (int k = 0; k <= in; k++) {
    for (int j = 0; j <= hid; j++) {
      input_weights_one_dim[m] = net->input_weights[k][j];
      input_weights_prev_one_dim[m] = net->input_prev_weights[k][j];
      m++;
    }
  }

  cudaMalloc((void**)&input_cuda, (in + 1) * sizeof(float));
  cudaMalloc((void**)&output_hidden_cuda, (hid + 1) * sizeof(float));
  cudaMalloc((void**)&input_hidden_cuda, (in + 1) * (hid + 1) * sizeof(float));
  cudaMalloc((void**)&hidden_partial_sum, num_blocks_b * WIDTH * sizeof(float));

#endif

#ifdef CPU

  printf("Performing CPU computation\n");
  bpnn_layerforward(net->input_units, net->hidden_units, net->input_weights, in,
                    hid);

#endif

#ifdef GPU

  printf("Performing GPU computation\n");

  // printf("in= %d, hid = %d, numblocks = %d\n", in, hid, num_blocks_b);

  cudaMemcpyAsync(input_cuda, net->input_units, (in + 1) * sizeof(float),
                  cudaMemcpyHostToDevice, stream_app);
  cudaMemcpyAsync(input_hidden_cuda, input_weights_one_dim,
                  (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice,
                  stream_app);

  bpnn_layerforward_CUDA<<<grid, threads, 0, stream_app>>>(
      input_cuda, output_hidden_cuda, input_hidden_cuda, hidden_partial_sum, in,
      hid);

  pthread_mutex_unlock(mutexapp);
  if (flag)
    cutilSafeCall(cudaStreamSynchronize(stream_app));
  else
    cutilSafeCall(cudaThreadSynchronize());

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("bpnn kernel error: %s\n", cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }

  cudaMemcpyAsync(partial_sum, hidden_partial_sum,
                  num_blocks_b * WIDTH * sizeof(float), cudaMemcpyDeviceToHost,
                  stream_app);

  for (int j = 1; j <= hid; j++) {
    sum = 0.0;
    for (int k = 0; k < num_blocks_b; k++) {
      sum += partial_sum[k * hid + j - 1];
    }
    sum += net->input_weights[0][j];
    net->hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
  }
#endif

  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights,
                    hid, out);
  bpnn_output_error(net->output_delta, net->target, net->output_units, out,
                    &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out,
                    net->hidden_weights, net->hidden_units, &hid_err);
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid,
                      net->hidden_weights, net->hidden_prev_weights);

#ifdef CPU

  bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in,
                      net->input_weights, net->input_prev_weights);

#endif

#ifdef GPU

  cudaMalloc((void**)&hidden_delta_cuda, (hid + 1) * sizeof(float));
  cudaMalloc((void**)&input_prev_weights_cuda,
             (in + 1) * (hid + 1) * sizeof(float));

  cudaMemcpyAsync(hidden_delta_cuda, net->hidden_delta,
                  (hid + 1) * sizeof(float), cudaMemcpyHostToDevice,
                  stream_app);
  cudaMemcpyAsync(input_prev_weights_cuda, input_weights_prev_one_dim,
                  (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice,
                  stream_app);
  cudaMemcpyAsync(input_hidden_cuda, input_weights_one_dim,
                  (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice,
                  stream_app);

  pthread_mutex_lock(mutexapp);
  bpnn_adjust_weights_cuda<<<grid, threads, 0, stream_app>>>(
      hidden_delta_cuda, hid, input_cuda, in, input_hidden_cuda,
      input_prev_weights_cuda);
  pthread_mutex_unlock(mutexapp);
  if (flag)
    cutilSafeCall(cudaStreamSynchronize(stream_app));
  cudaMemcpyAsync(net->input_units, input_cuda, (in + 1) * sizeof(float),
                  cudaMemcpyDeviceToHost, stream_app);
  cudaMemcpyAsync(input_weights_one_dim, input_hidden_cuda,
                  (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyDeviceToHost,
                  stream_app);
  if (flag)
    cutilSafeCall(cudaStreamSynchronize(stream_app));
  cudaFree(input_cuda);
  cudaFree(output_hidden_cuda);
  cudaFree(input_hidden_cuda);
  cudaFree(hidden_partial_sum);
  cudaFree(input_prev_weights_cuda);
  cudaFree(hidden_delta_cuda);

  free(partial_sum);
  free(input_weights_one_dim);
  free(input_weights_prev_one_dim);

#endif
}
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
// int
// main( int argc, char** argv)
int main_BP(cudaStream_t stream_app, pthread_mutex_t* mutexapp, bool flag) {
  int seed;
  int layer_size = 65536;
  if (layer_size % 16 != 0) {
    fprintf(stderr, "The number of input points must be divided by 16\n");
    exit(0);
  }

  seed = 7;
  bpnn_initialize(seed);
  BPNN* net;
  float out_err, hid_err;
  net = bpnn_create(layer_size, 16, 1);  // (16, 1 can not be changed)

  printf("Input layer size : %d\n", layer_size);
  // load(net);
  float* units;
  int nr, k;

  nr = layer_size;

  // imgsize = nr * nc;
  units = net->input_units;

  k = 1;
  for (int i = 0; i < nr; i++) {
    units[k] = (float)rand() / RAND_MAX;
    k++;
  }
  // entering the training kernel, only one iteration
  printf("Starting training kernel\n");
  bpnn_train_cuda(net, &out_err, &hid_err, stream_app, mutexapp, flag);
  bpnn_free(net);
  printf("Training done\n");
  return 0;
}
