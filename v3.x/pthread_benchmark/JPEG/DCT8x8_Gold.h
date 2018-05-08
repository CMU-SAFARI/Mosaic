/*
* Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
*
* NVIDIA Corporation and its licensors retain all intellectual property and 
* proprietary rights in and to this software and related documentation and 
* any modifications thereto.  Any use, reproduction, disclosure, or distribution 
* of this software and related documentation without an express license 
* agreement from NVIDIA Corporation is strictly prohibited.
* 
*/

/**
**************************************************************************
* \file DCT8x8_Gold.h
* \brief Contains declaration of CPU versions of DCT, IDCT and quantization 
* routines.
*
* Contains declaration of CPU versions of DCT, IDCT and quantization 
* routines.
*/


#pragma once

#include "BmpUtil.h"

extern "C" 
{
	void computeDCT8x8Gold1(const float* fSrc, float* fDst, int Stride, ROI Size);
	void computeIDCT8x8Gold1(const float* fSrc, float* fDst, int Stride, ROI Size);
	void quantizeGoldFloat(float* fSrcDst, int Stride, ROI Size);
	void quantizeGoldShort(short* fSrcDst, int Stride, ROI Size);
	void computeDCT8x8Gold2(const float* fSrc, float* fDst, int Stride, ROI Size);
	void computeIDCT8x8Gold2(const float* fSrc, float* fDst, int Stride, ROI Size);
}
