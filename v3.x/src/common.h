#ifndef COMMON_H_INCLUDED
#define COMMON_H_INCLUDED

#include <pthread.h>

struct thread_to_appID_struct {
	int size;
	void** threadIDs;
	int* appIDs;
	void init() {
		size = 0;
		threadIDs = NULL;
		appIDs = NULL;
	}
	int lookup(void* threadID) {
		for (int i = 0; i < size; ++i) {
			if (threadIDs[i] == threadID) return appIDs[i];
		}
		return -1;
	}
	void add(void* threadID, int appID) {
		if (lookup(threadID) != -1) return;
		++size;
		threadIDs = (void**)realloc(threadIDs, size*sizeof(void*));
		appIDs = (int*)realloc(appIDs, size*sizeof(int));
		threadIDs[size-1] = threadID;
		appIDs[size-1] = appID;
	}
};
extern thread_to_appID_struct* thread_to_appID;

#endif