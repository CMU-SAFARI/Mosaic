#ifndef _LIBDATA_H_
#define _LIBDATA_H_
#include <math.h>
#include "qtclib.h"
#include "support.h"

float* generate_synthetic_data(float** rslt_mtrx,
                               int** indr_mtrx,
                               int* max_degree,
                               float threshold,
                               int N,
                               int type);

#endif
