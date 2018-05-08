#include "CudaRandomAccess.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cutil_inline.h>
#include "../benchmark_common.h"

/* Number of updates to table (suggested: 4x number of table entries) */
#define UP_REPEAT 4
#define NUPDATE (UP_REPEAT * TableSize)
#define THREAD_BLOCK 128
#define NTHREAD_BLOCKS 512
#define MAX_VL 128

inline double RTSEC()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return ((double) tv.tv_sec + ((double) tv.tv_usec) / 1000000);
}


//int main()
int main_gups(cudaStream_t stream_app, pthread_mutex_t *mutexapp, bool flag) 
{

  Params_t params;
  double GUPs;
  int failure;
  cudaSetDevice( cutGetMaxGflopsDeviceId() );
  params.HPLMaxProcMem = TOTAL_MEM;
  params.outFname = "RA_output";
  HPCC_RandomAccess(&params, 0, &GUPs, &failure, stream_app, mutexapp,flag);

  return 0;
}


/* Utility routine to start random number generator at Nth step */
u64Int HPCC_starts(s64Int n)
{
  int i, j;
  u64Int m2[64];
  u64Int temp, ran;

  while (n < 0) n += PERIOD;
  while (n > PERIOD) n -= PERIOD;
  if (n == 0) return 0x1;

  temp = 0x1;
  for (i=0; i<64; i++) {
    m2[i] = temp;
    temp = (temp << 1) ^ ((s64Int) temp < 0 ? POLY : 0);
    temp = (temp << 1) ^ ((s64Int) temp < 0 ? POLY : 0);
  }

  for (i=62; i>=0; i--)
    if ((n >> i) & 1)
      break;

  ran = 0x2;
  while (i > 0) {
    temp = 0;
    for (j=0; j<64; j++)
      if ((ran >> j) & 1)
        temp ^= m2[j];
    ran = temp;
    i -= 1;
    if ((n >> i) & 1)
      ran = (ran << 1) ^ ((s64Int) ran < 0 ? POLY : 0);
  }

  return ran;
}


#ifdef CUDA_VERSION
// trick to let us access the 64-bit simulator xor intrinsic
#define NF_ATOMIC
#ifdef NF_ATOMIC
extern "C" __device__ __noinline__ void _Z_intrinsic_atom_global_xor_nf_i64(void* address, unsigned long long int data) {
    atomicXor(0 + (unsigned int *) address, * (0 + (unsigned int *) & data));
    atomicXor(1 + (unsigned int *) address, * (1 + (unsigned int *) & data));
}
#else
extern "C" __device__ __noinline__ unsigned long long int _Z_intrinsic_atom_global_xor_i64(void* address, unsigned long long int data) {
    unsigned long long int previous = *(unsigned long long int*)address;
    atomicXor(0 + (unsigned int *) address, * (0 + (unsigned int *) & data));
    atomicXor(1 + (unsigned int *) address, * (1 + (unsigned int *) & data));
    return previous;
}
#endif

//CUDA kernel
__global__ static void RandomAccessUpdate(u64Int TableSize, u64Int *Table, u64Int *starts) {
  u64Int i;
  u64Int ran;              /* Current random number */

  /* Perform updates to main table.  The scalar equivalent is:
   *
   *     u64Int ran;
   *     ran = 1;
   *     for (i=0; i<NUPDATE; i++) {
   *       ran = (ran << 1) ^ (((s64Int) ran < 0) ? POLY : 0);
   *       table[ran & (TableSize-1)] ^= ran;
   *     }
   */
  // ran = HPCC_starts ((NUPDATE/THREAD_BLOCK/NTHREAD_BLOCKS) * (blockIdx.x * blockDim.x + threadIdx.x));
  ran = starts[blockIdx.x * blockDim.x + threadIdx.x];

  for (i=0; i<NUPDATE/THREAD_BLOCK/NTHREAD_BLOCKS; i++) {
    ran = (ran << 1) ^ ((s64Int) ran < 0 ? POLY : 0);
    u64Int index = ran & (TableSize-1);
#ifdef ATOMIC_64_BIT_XOR
#ifdef ATOMIC_INTRINSIC
#ifdef NF_ATOMIC
    _Z_intrinsic_atom_global_xor_nf_i64(&Table[index], ran);
#else
    _Z_intrinsic_atom_global_xor_i64(&Table[index], ran);
#endif
#else
    atomicXor(&Table[index], ran);
#endif
#else
    atomicXor(0 + (unsigned int *) &Table[index], * (0 + (unsigned int *) & ran));
    atomicXor(1 + (unsigned int *) &Table[index], * (1 + (unsigned int *) & ran));
#endif
  }
}
#else
#ifdef C_VECTOR_VERSION
// MAX_VL element vector version
static void RandomAccessUpdate(u64Int TableSize, u64Int *Table) {
  u64Int i;
  u64Int ran[MAX_VL];              /* Current random numbers */
  int j;

  /* Perform updates to main table.  The scalar equivalent is:
   *
   *     u64Int ran;
   *     ran = 1;
   *     for (i=0; i<NUPDATE; i++) {
   *       ran = (ran << 1) ^ (((s64Int) ran < 0) ? POLY : 0);
   *       table[ran & (TableSize-1)] ^= ran;
   *     }
   */
  for (j=0; j<MAX_VL; j++)
    ran[j] = HPCC_starts ((NUPDATE/MAX_VL) * j);

  for (i=0; i<NUPDATE/MAX_VL; i++) {
    for (j=0; j<MAX_VL; j++) {
      ran[j] = (ran[j] << 1) ^ ((s64Int) ran[j] < 0 ? POLY : 0);
      Table[ran[j] & (TableSize-1)] ^= ran[j];
    }
  }
}
#else
// strict scalar version
static void RandomAccessUpdate(u64Int TableSize, u64Int *Table) {
  u64Int i;
  u64Int ran;              /* Current random number */

  /* Perform updates to main table.  The scalar equivalent is:
   *
   *     u64Int ran;
   *     ran = 1;
   *     for (i=0; i<NUPDATE; i++) {
   *       ran = (ran << 1) ^ (((s64Int) ran < 0) ? POLY : 0);
   *       table[ran & (TableSize-1)] ^= ran;
   *     }
   */
  ran = 0x1;

  for (i=0; i<NUPDATE; i++) {
    ran = (ran << 1) ^ ((s64Int) ran < 0 ? POLY : 0);
    Table[ran & (TableSize-1)] ^= ran;
  }
}
#endif
#endif

int
HPCC_RandomAccess(Params_t *params, int doIO, double *GUPs, int *failure, cudaStream_t stream_app, pthread_mutex_t *mutexapp, bool flag) {
  u64Int i;
  u64Int temp;
  double cputime;               /* CPU time to update table */
  double realtime;              /* Real time to update table */
  double totalMem;
  u64Int *Table;
  u64Int logTableSize, TableSize;
  FILE *outFile = NULL;

  if (doIO) {
    outFile = fopen( params->outFname, "a" );
    if (! outFile) {
      outFile = stderr;
      fprintf( outFile, "Cannot open output file.\n" );
      return 1;
    }
  }

  /* calculate local memory per node for the update table */
  totalMem = params->HPLMaxProcMem;
  totalMem /= sizeof(u64Int);

  /* calculate the size of update array (must be a power of 2) */
  for (totalMem *= 0.5, logTableSize = 0, TableSize = 1;
       totalMem >= 1.0;
       totalMem *= 0.5, logTableSize++, TableSize <<= 1)
    ; /* EMPTY */


  Table = (typeof(Table)) malloc (sizeof(u64Int) * TableSize);
  if (! Table) {
    if (doIO) {
      fprintf( outFile, "Failed to allocate memory for the update table (" FSTR64 ").\n", TableSize);
      fclose( outFile );
    }
    return 1;
  }

  /* Print parameters for run */
  if (doIO) {
  fprintf( outFile, "------------------------------------------------------------\n");
#ifdef CUDA_VERSION
  fprintf( outFile, "GPU version\n");
  fprintf( outFile, "CTA_SIZE = %d, CTAS = %d\n", THREAD_BLOCK, NTHREAD_BLOCKS);
  cudaDeviceProp deviceProp;
  cutilSafeCall(cudaGetDeviceProperties(&deviceProp, 0));
  fprintf( outFile, "Device Name = %s\n", deviceProp.name);
#else
#  ifdef C_VECTOR_VERSION
  fprintf( outFile, "CPU Vector version\n");
  fprintf( outFile, "Vector length\n");
#  else
  fprintf( outFile, "CPU non-Vector version\n");
#  endif
#endif
  fprintf( outFile, "Main table size   = 2^" FSTR64 " = " FSTR64 " words\n", logTableSize,TableSize);
  fprintf( outFile, "Number of updates = " FSTR64 "\n", NUPDATE);
  }

  /* Initialize main table */
  for (i=0; i<TableSize; i++) Table[i] = i;
#ifdef CUDA_VERSION
  u64Int starts[THREAD_BLOCK*NTHREAD_BLOCKS];
  for (i=0; i< (THREAD_BLOCK*NTHREAD_BLOCKS); i++) starts[i] = HPCC_starts ((NUPDATE/THREAD_BLOCK/NTHREAD_BLOCKS) * i);
#endif


  /* Begin timing here */
#if 0
  cputime = -CPUSEC();
#else
  cputime = 0;
#endif
  realtime = -RTSEC();

#ifdef CUDA_VERSION
  int size = sizeof(u64Int) * TableSize;
  u64Int *d_Table;
  u64Int *d_starts;
  cutilSafeCall(cudaMalloc(&d_Table, size));
printf("sizeof(starts) = %lld", sizeof(starts));
  cutilSafeCall(cudaMalloc(&d_starts, sizeof(starts)));
  cutilSafeCall(cudaMemcpyAsync(d_Table, Table, size, cudaMemcpyHostToDevice, stream_app));
  cutilSafeCall(cudaMemcpyAsync(d_starts, starts, sizeof(starts), cudaMemcpyHostToDevice, stream_app));
  RandomAccessUpdate <<<NTHREAD_BLOCKS, THREAD_BLOCK, 0, stream_app>>> (TableSize, d_Table, d_starts);


  cutilSafeCall(cudaGetLastError());
  cutilSafeCall(cudaMemcpyAsync(Table, d_Table, size, cudaMemcpyDeviceToHost, stream_app));
#else
  RandomAccessUpdate( TableSize, Table );
#endif
  pthread_mutex_unlock (mutexapp);
  //cutilSafeCall( cudaStreamSynchronize(stream_app) );
  if(flag)
      cutilSafeCall( cudaStreamSynchronize(stream_app) );


#if 0
  /* End timed section */
  cputime += CPUSEC();
#endif
  realtime += RTSEC();

  /* make sure no division by zero */
  *GUPs = (realtime > 0.0 ? 1.0 / realtime : -1.0);
  *GUPs *= 1e-9*NUPDATE;
  /* Print timing results */
  if (doIO) {
  fprintf( outFile, "CPU time used  = %.6f seconds\n", cputime);
  fprintf( outFile, "Real time used = %.6f seconds\n", realtime);
  fprintf( outFile, "%.9f Billion(10^9) Updates    per second [GUP/s]\n", *GUPs );
  }

  /* Verification of results (in serial or "safe" mode; optional) */
  temp = 0x1;
  for (i=0; i<NUPDATE; i++) {
    temp = (temp << 1) ^ (((s64Int) temp < 0) ? POLY : 0);
    Table[temp & (TableSize-1)] ^= temp;
  }

  temp = 0;
  for (i=0; i<TableSize; i++)
    if (Table[i] != i)
      temp++;

  if (doIO) {
  fprintf( outFile, "Found " FSTR64 " errors in " FSTR64 " locations (%s).\n",
           temp, TableSize, (temp <= 0.01*TableSize) ? "passed" : "failed");
  }
  if (temp <= 0.01*TableSize) *failure = 0;
  else *failure = 1;

  free( Table );
#ifdef CUDA_VERSION
  cutilSafeCall(cudaFree (d_Table));
#endif

  if (doIO) {
    fflush( outFile );
    fclose( outFile );
  }

  return 0;
}
