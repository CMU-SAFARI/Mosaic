#ifndef SCAN_H_
#define SCAN_H_

template <class T>
bool scanCPU(T *data, T* reference, T* dev_result, const size_t size);

template <class T, class vecT>
void RunTest(cudaStream_t stream_app, pthread_mutex_t *mutexapp, bool flag);

#endif // SCAN_H_
