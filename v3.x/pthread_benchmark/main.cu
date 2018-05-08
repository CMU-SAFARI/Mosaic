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

#include "../src/gpgpu-sim/ConfigOptions.h"
#include "../src/gpgpu-sim/App.h"
#include "../src/common.h"

thread_to_appID_struct* thread_to_appID;

struct app_data {
  app_data(char* app_name, pthread_mutex_t* app_mutex, bool concurrent, cudaEvent_t* done,
      std::vector<cudaEvent_t>* done_events, size_t app_num) :
      app_name(app_name), app_mutex(app_mutex), concurrent(concurrent), done(done),
      done_events(done_events), appID(app_num) {
    cutilSafeCall(cudaStreamCreate(&stream));
  }
  cudaStream_t stream;
  cudaEvent_t* done;
  std::vector<cudaEvent_t>* done_events;
  char* app_name;
  pthread_mutex_t* app_mutex;
  bool concurrent;
  size_t appID;
};

int callApp(char *app_name, cudaStream_t stream_app, pthread_mutex_t *mutexapp, bool concurrent) {
  if (strcmp(app_name, "NN") == 0)
    main_NN(stream_app, mutexapp, concurrent);
  else if (strcmp(app_name, "BP") == 0)
    main_BP(stream_app, mutexapp, concurrent);
  else if (strcmp(app_name, "FFT") == 0)
    main_fft(stream_app, mutexapp, concurrent);
//  else if(strcmp(app_name,"MUM") ==0)
//    main_MUM(stream_app, mutexapp, concurrent);
  else if (strcmp(app_name, "LUH") == 0)
    main_lulesh(stream_app, mutexapp, concurrent);
  else if (strcmp(app_name, "RED") == 0)
    main_RED(stream_app, mutexapp, concurrent);
  else if (strcmp(app_name, "SCAN") == 0)
    main_scan(stream_app, mutexapp, concurrent);
  else if (strcmp(app_name, "CFD") == 0)
    main_cfd(stream_app, mutexapp, concurrent);
  else if (strcmp(app_name, "TRD") == 0)
    main_TRD(stream_app, mutexapp, concurrent);
  else if (strcmp(app_name, "SPMV") == 0)
    main_spmv(stream_app, mutexapp, concurrent);
  else if (strcmp(app_name, "NW") == 0)
    main_nw(stream_app, mutexapp, concurrent);
  else if (strcmp(app_name, "3DS") == 0)
    main_threeDS(stream_app, mutexapp, concurrent);
  else if (strcmp(app_name, "HS") == 0)
    main_hotspot(stream_app, mutexapp, concurrent);
  else if (strcmp(app_name, "HISTO") == 0)
    main_histo(stream_app, mutexapp, concurrent);
  else if (strcmp(app_name, "SC") == 0)
    main_streamcluster(stream_app, mutexapp, concurrent);
  else if (strcmp(app_name, "SCP") == 0)
    main_SCP(stream_app, mutexapp, concurrent);
  else if (strcmp(app_name, "GUPS") == 0)
    main_gups(stream_app, mutexapp, concurrent);
  else if (strcmp(app_name, "QTC") == 0)
    main_QTC(stream_app, mutexapp, concurrent);
  else if (strcmp(app_name, "LUD") == 0)
    main_LUD(stream_app, mutexapp, concurrent);
  else if (strcmp(app_name, "SRAD") == 0)
    main_SRAD(stream_app, mutexapp, concurrent);
  else if (strcmp(app_name, "CONS") == 0)
    main_CONS(stream_app, mutexapp, concurrent);
  else if (strcmp(app_name, "SAD") == 0)
    main_sad(stream_app, mutexapp, concurrent);
  else if (strcmp(app_name, "MM") == 0)
    main_MM(stream_app, mutexapp, concurrent);
  else if (strcmp(app_name, "JPEG") == 0)
    main_JPEG(stream_app, mutexapp, concurrent);
  else if (strcmp(app_name, "BFS2") == 0)
    main_BFS2(stream_app, mutexapp, concurrent);
  else if (strcmp(app_name, "FWT") == 0)
    main_FWT(stream_app, mutexapp, concurrent);
  else if (strcmp(app_name, "LPS") == 0)
    main_LPS(stream_app, mutexapp, concurrent);
  else if (strcmp(app_name, "LIB") == 0)
    main_lib(stream_app, mutexapp, concurrent);
  else if (strcmp(app_name, "RAY") == 0)
    main_ray(stream_app, mutexapp, concurrent);
  else if (strcmp(app_name, "BLK") == 0)
    main_BlackScholes(stream_app, mutexapp, concurrent);
  else
    return EXIT_FAILURE;
  return EXIT_SUCCESS;
}

void* benchmark(void *app_arg) {
  struct app_data* app = (struct app_data*) app_arg;
  char* name = app->app_name;
  int ret;
  bool some_running = false;
  do {
    pthread_mutex_lock(app->app_mutex);
    printf("Launch code in main.cu:launching a new benchmark, appID = %d, already registered? = %d\n", app->appID, App::is_registered(app->appID));
    if(App::is_registered(app->appID)) thread_to_appID->add((void*)pthread_self(), App::get_app_id(app->appID));
    else thread_to_appID->add((void*)pthread_self(), App::register_app(app->appID));
    //thread_to_appID->add((void*)pthread_self(), app->appID);
    ret = callApp(name, app->stream, app->app_mutex, app->concurrent);
    if (ret) {
      fprintf(stderr, "Error launching benchmark %s\n", name);
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

void usage(char* const name) {
  printf("Usage: %s app1 app2 app3 .. appN\n", name);
}

int main(int argc, char *argv[]) {
  if (argc == 1) {
    usage(argv[0]);
    exit(EXIT_FAILURE);
  }
  size_t n_apps = argc - 1;
  std::vector<app_data> apps;
  std::vector<void*> status(n_apps, NULL);
  std::vector<pthread_t> threads(n_apps, 0);
  std::vector<cudaEvent_t> done_events(n_apps, 0);
  pthread_mutex_t app_mutex;

  // set global state hack
  ConfigOptions::n_apps = n_apps;
  pthread_mutex_init(&app_mutex, NULL);
  thread_to_appID = (thread_to_appID_struct *)malloc(sizeof(thread_to_appID_struct));
  thread_to_appID->init();

  bool concurrent = n_apps > 1;
  for (size_t i = 0; i < n_apps; i++) {
    cutilSafeCall(cudaEventCreate(&done_events[i]));
    apps.push_back(app_data(argv[i + 1], &app_mutex, concurrent, &done_events[i], &done_events, i));
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
