//#include <stdint.h>

/* Random number generator */
#ifdef LONG_IS_64BITS
#define POLY 0x0000000000000007UL
#define PERIOD 1317624576693539401L
#else
#define POLY 0x0000000000000007ULL
#define PERIOD 1317624576693539401LL
#endif

#ifdef LONG_IS_64BITS
#define FSTR64 "%ld"
#define FSTRU64 "%lu"
#define ZERO64B 0L
#else
#define FSTR64 "%lld"
#define FSTRU64 "%llu"
#define ZERO64B 0LL
#endif

#define TOTAL_MEM (1 << 22);


//typedef uint64_t u64Int;
//typedef int64_t s64Int;
typedef unsigned long long u64Int;
typedef long long s64Int;

typedef struct {
  char *outFname;
  u64Int HPLMaxProcMem;
} Params_t;

#if 0
/* Macros for timing */
#define CPUSEC() (HPL_timer_cputime())
#define RTSEC() (MPI_Wtime())
#endif

extern u64Int HPCC_starts(s64Int);

extern u64Int *HPCC_Table;

extern int HPCC_RandomAccess(Params_t *params, int doIO, double *GUPs, int *failure, cudaStream_t stream_app, pthread_mutex_t *mutexapp, bool flag);

