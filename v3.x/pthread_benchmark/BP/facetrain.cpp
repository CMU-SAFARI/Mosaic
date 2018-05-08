

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include "backprop.h"
#include "omp.h"
#include "../benchmark_common.h"
extern char *strcpy();
//extern void exit();

int layer_size = 0;

void backprop_face(cudaStream_t stream_app, pthread_mutex_t *mutexapp, bool flag)
{
  BPNN *net;
  int i;
  float out_err, hid_err;
  net = bpnn_create(layer_size, 16, 1); // (16, 1 can not be changed)
  
  printf("Input layer size : %d\n", layer_size);
  //load(net);
    float *units;
    int nr, nc, imgsize, j, k;
    
    nr = layer_size;
    
    imgsize = nr * nc;
    units = net->input_units;
    
    k = 1;
    for (int i = 0; i < nr; i++) {
        units[k] = (float) rand()/RAND_MAX ;
        k++;
    }
  //entering the training kernel, only one iteration
  printf("Starting training kernel\n");
  bpnn_train_cuda(net, &out_err, &hid_err, stream_app, mutexapp, flag);
  bpnn_free(net);
  printf("Training done\n");
  //return 0;
}

int setup(cudaStream_t stream_app, pthread_mutex_t *mutexapp, bool flag)
//int argc;
//char *argv[];
{
	
  int seed;

  /*if (argc!=2){
  fprintf(stderr, "usage: backprop <num of input elements>\n");
  exit(0);
  }*/
  layer_size = 65536;
  if (layer_size%16!=0){
  fprintf(stderr, "The number of input points must be divided by 16\n");
  exit(0);
  }
  

  seed = 7;   
  bpnn_initialize(seed);
  backprop_face(stream_app, mutexapp, flag);

  //exit(0);
  return 0;
}
