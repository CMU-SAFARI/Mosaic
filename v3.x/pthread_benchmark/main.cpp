#include <argp.h>
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cutil_inline.h>
#include <unistd.h>
#include "benchmark_common.h"
#include <iomanip>
#include <vector>

#include "ConfigOptions.h"
#include "App.h"

struct app_data {
  app_data(char* app_name, pthread_mutex_t* app_mutex, bool concurrent,
      cudaEvent_t* done, std::vector<cudaEvent_t>* done_events, size_t app_num) :
      app_name(app_name), mutex(app_mutex), concurrent(concurrent),
      done(done), done_events(done_events), appID(app_num) {
    cutilSafeCall(cudaStreamCreate(&stream));
  }
  cudaStream_t stream;
  cudaEvent_t* done;
  std::vector<cudaEvent_t>* done_events;
  char* app_name;
  pthread_mutex_t* mutex;
  bool concurrent;
  size_t appID;
};

int callApp(struct app_data* app) {
  if (strcmp(app->app_name, "NN") == 0)
    main_NN(app->stream, app->mutex, app->concurrent);
  else if (strcmp(app->app_name, "BP") == 0)
    main_BP(app->stream, app->mutex, app->concurrent);
  else if (strcmp(app->app_name, "FFT") == 0)
    main_fft(app->stream, app->mutex, app->concurrent);
//  else if(strcmp(app->app_name,"MUM") == 0)
//    main_MUM(app->stream, app->mutex, app->concurrent);
  else if (strcmp(app->app_name, "LUH") == 0)
    main_lulesh(app->stream, app->mutex, app->concurrent);
  else if (strcmp(app->app_name, "RED") == 0)
    main_RED(app->stream, app->mutex, app->concurrent);
  else if (strcmp(app->app_name, "SCAN") == 0)
    main_scan(app->stream, app->mutex, app->concurrent);
  else if (strcmp(app->app_name, "CFD") == 0)
    main_cfd(app->stream, app->mutex, app->concurrent);
  else if (strcmp(app->app_name, "TRD") == 0)
    main_TRD(app->stream, app->mutex, app->concurrent);
  else if (strcmp(app->app_name, "SPMV") == 0)
    main_spmv(app->stream, app->mutex, app->concurrent);
  else if (strcmp(app->app_name, "NW") == 0)
    main_nw(app->stream, app->mutex, app->concurrent);
  else if (strcmp(app->app_name, "3DS") == 0)
    main_threeDS(app->stream, app->mutex, app->concurrent);
  else if (strcmp(app->app_name, "HS") == 0)
    main_hotspot(app->stream, app->mutex, app->concurrent);
  else if (strcmp(app->app_name, "HISTO") == 0)
    main_histo(app->stream, app->mutex, app->concurrent);
  else if (strcmp(app->app_name, "SC") == 0)
    main_streamcluster(app->stream, app->mutex, app->concurrent);
  else if (strcmp(app->app_name, "SCP") == 0)
    main_SCP(app->stream, app->mutex, app->concurrent);
  else if (strcmp(app->app_name, "GUPS") == 0)
    main_gups(app->stream, app->mutex, app->concurrent);
  else if (strcmp(app->app_name, "QTC") == 0)
    main_QTC(app->stream, app->mutex, app->concurrent);
  else if (strcmp(app->app_name, "LUD") == 0)
    main_LUD(app->stream, app->mutex, app->concurrent);
  else if (strcmp(app->app_name, "SRAD") == 0)
    main_SRAD(app->stream, app->mutex, app->concurrent);
  else if (strcmp(app->app_name, "CONS") == 0)
    main_CONS(app->stream, app->mutex, app->concurrent);
  else if (strcmp(app->app_name, "SAD") == 0)
    main_sad(app->stream, app->mutex, app->concurrent);
  else if (strcmp(app->app_name, "MM") == 0)
    main_MM(app->stream, app->mutex, app->concurrent);
  else if (strcmp(app->app_name, "JPEG") == 0)
    main_JPEG(app->stream, app->mutex, app->concurrent);
  else if (strcmp(app->app_name, "BFS2") == 0)
    main_BFS2(app->stream, app->mutex, app->concurrent);
  else if (strcmp(app->app_name, "FWT") == 0)
    main_FWT(app->stream, app->mutex, app->concurrent);
  else if (strcmp(app->app_name, "LPS") == 0)
    main_LPS(app->stream, app->mutex, app->concurrent);
  else if (strcmp(app->app_name, "LIB") == 0)
    main_lib(app->stream, app->mutex, app->concurrent);
  else if (strcmp(app->app_name, "RAY") == 0)
    main_ray(app->stream, app->mutex, app->concurrent);
  else if (strcmp(app->app_name, "BLK") == 0)
    main_BlackScholes(app->stream, app->mutex, app->concurrent);
  else
    return EXIT_FAILURE;
  return EXIT_SUCCESS;
}

/**
 * To be run from pthread_create
 */
void* benchmark(void *app_arg) {
  struct app_data* app = (struct app_data*) app_arg;
  int ret;
  bool some_running = false;
  do {
    pthread_mutex_lock(app->mutex);
    printf("Launch code in main.cu:launching a new benchmark, appID = %s\n", app->appID);
    assert(!App::is_registered(app->appID));
    App::register_app(app->appID);
    ret = callApp(app);
    if (ret) {
      fprintf(stderr, "Error launching benchmark %s\n", app->app_name);
    }
    cutilSafeCall(cudaEventRecord(*app->done, app->stream));
    // Loop until all apps have completed one iteration
    for (std::vector<cudaEvent_t>::iterator e = app->done_events->begin();
        e != app->done_events->end(); e++) {
      if (cudaEventQuery(*e) == cudaErrorNotReady) {
        some_running = true;
      }
    }
  } while (some_running);

  pthread_exit((void*) ret);
  return NULL;
}

/**
 * Command line argument setup
 */
static char doc[] = "GPGPU-Sim Launcher";
static char args_doc[] = "Specify applications and paired application priorities with -a and -p.";

static struct argp_option options[] = {
  {"app", 'a', "APP_NAME", 0, "Application to run"},
  {0},
};

struct arguments {
  std::vector<char*> apps;
  std::vector<int> priorities;
};

static error_t parse_opt(int key, char* arg, struct argp_state* state) {
  struct arguments* args = (struct arguments*) state->input;
  switch (key) {
    case 'a':
      args->apps.push_back(arg);
      break;
    case 'p':
      args->priorities.push_back((int) strtol(arg, NULL, 0));
      break;
    default:
      return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

static struct argp argp = {options, parse_opt, args_doc, doc};

int main(int argc, char *argv[]) {
  struct arguments args;
  argp_parse(&argp, argc, argv, 0, 0, &args);
  size_t n_apps = args.apps.size();
  if (!n_apps) {
    fprintf(stderr, "ERROR: %s %s\n", argv[0], argp.args_doc);
    exit(EXIT_FAILURE);
  }
  std::vector<app_data> apps;
  std::vector<void*> status(n_apps, NULL);
  std::vector<pthread_t> threads(n_apps, 0);
  std::vector<cudaEvent_t> done_events(n_apps, 0);
  pthread_mutex_t app_mutex;

  // set global state hack
  ConfigOptions::n_apps = n_apps;

  pthread_mutex_init(&app_mutex, NULL);

  bool concurrent = n_apps > 1;
  for (size_t i = 0; i < n_apps; i++) {
    cutilSafeCall(cudaEventCreate(&done_events[i]));
    apps.push_back(app_data(args.apps[i], &app_mutex, concurrent,
        &done_events[i], &done_events, i));
  }
  // Launch benchmark threads
  for (size_t i = 0; i < n_apps; i++) {
    errno = pthread_create(&threads[i], NULL, benchmark, &apps[i]);
    if (errno) {
      fprintf(stderr, "Error creating thread: %s\n", strerror(errno));
      exit(EXIT_FAILURE);
    }
    sleep(1); // TODO keep this?
  }

  // Wait for completion
  for (size_t i = 0; i < n_apps; i++) {
    errno = pthread_join(threads[i], &status[i]);
    if (errno) {
      fprintf(stderr, "Error creating thread: %s\n", strerror(errno));
      exit(EXIT_FAILURE);
    }
    printf("Main: completed join with thread %ud having a status of %ld\n", i, (long) status[i]);
  }

  // Clean up
  for (size_t i = 0; i < n_apps; i++) {
    cutilSafeCall(cudaStreamDestroy(apps[i].stream));
  }

  pthread_mutex_destroy(&app_mutex);
  printf("Main: program completed. Exiting.\n");

  return 0;
}
