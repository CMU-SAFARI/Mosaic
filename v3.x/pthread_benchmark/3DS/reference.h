#ifndef _MP_3DFD_VERIFICATION
#define _MP_3DFD_VERIFICATION

#include <stdio.h>

void init_data(float *data, const int dimx, const int dimy, const int dimz)
{
	for(int iz=0; iz<dimz; iz++)
		for(int iy=0; iy<dimy; iy++)
			for(int ix=0; ix<dimx; ix++)
			{
				*data = (float)iz;
				++data;
			}
}

void random_data(float *data, const int dimx, const int dimy, const int dimz, const int lower_bound, const int upper_bound)
{
	srand(0);

	for(int iz=0; iz<dimz; iz++)
		for(int iy=0; iy<dimy; iy++)
			for(int ix=0; ix<dimx; ix++)
			{
				*data = 1.f;//(float) (lower_bound + (rand() % (upper_bound - lower_bound)));
				++data;
			}
}

// note that this CPU implemenation is extremely naive and slow, NOT to be used for performance comparisons
void reference_3D(float *output, float *input, float *coeff, const int dimx, const int dimy, const int dimz, const int radius=4)
{
	int dimxy = dimx*dimy;

	for(int iz=0; iz<dimz; iz++)
	{
		for(int iy=0; iy<dimy; iy++)
		{
			for(int ix=0; ix<dimx; ix++)
			{
				if( ix>=radius && ix<(dimx-radius) && iy>=radius && iy<(dimy-radius) && iz>=radius && iz<(dimz-radius) )
				{
					float value = (*input)*coeff[0];

					for(int ir=1; ir<=radius; ir++)
					{
						value += coeff[ir] * (*(input+ir) + *(input-ir));				// horizontal
						value += coeff[ir] * (*(input+ir*dimx) + *(input-ir*dimx));		// vertical
						value += coeff[ir] * (*(input+ir*dimxy) + *(input-ir*dimxy));	// in front / behind
					}

					*output = value;
				}

				++output;
				++input;
			}
		}
	}
}

bool within_delta(float* output, float *reference, 
				  const int nx, const int ny, const int nz,
				  const int dimx, const int dimy, const int dimz, 
				  const int radius, const float delta=0.0001f )
{
	bool retval = true;

	for(int iz=0; iz<dimz; iz++)
	{
		for(int iy=0; iy<dimy; iy++)
		{
			for(int ix=0; ix<dimx; ix++)
			{
				if( ix>=radius && ix<(nx+radius) && iy>=radius && iy<(dimy-radius) && iz>=radius && iz<(dimz-radius) )
				{
					float difference = abs( *reference - *output);

					if( difference > delta )
					{
						retval = false;
						printf(" ERROR: (%d,%d,%d)\t%.2f instead of %.2f\n", ix,iy,iz, *output, *reference);
						
						return false;
					}
					//printf("%d %d %.2f\n", ix,iy, difference);
				}

				++output;
				++reference;
			}
		}
	}

	return retval;
}

size_t num_errors(float* output, float *reference, const int dimx, const int dimy, const int dimz, const int radius=4, const int zadjust=-1, const float delta=0.0001f )
{
	size_t retval = 0;

	for(int iz=0; iz<dimz; iz++)
	{
		for(int iy=0; iy<dimy; iy++)
		{
			for(int ix=0; ix<dimx; ix++)
			{
				if( ix>=radius && ix<(dimx-radius) && iy>=radius && iy<(dimy-radius) && iz>=radius && iz<(dimz-radius) )
				{
					float difference = abs( *reference - *output);

					if( difference > delta )
					{
						retval++;
					}
				}

				++output;
				++reference;
			}
		}
	}

	return retval;
}

#endif
