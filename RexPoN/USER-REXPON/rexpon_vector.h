/*----------------------------------------------------------------------
  This program is free software; you can redistribute it and/or
  modify it under the terms of the GNU General Public License as
  published by the Free Software Foundation; either version 2 of
  the License, or (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  See the GNU General Public License for more details:
  <http://www.gnu.org/licenses/>.
  ----------------------------------------------------------------------*/

#ifndef __VECTOR_H_
#define __VECTOR_H_

#include "pair.h"
#include "rexpon_types.h"
#include "rexpon_defs.h"

void rvec_Copy( rvec, rvec );
void rvec_Scale( rvec, double, rvec );
void rvec_Add( rvec, rvec );
void rvec_ScaledAdd( rvec, double, rvec );
void rvec_ScaledSum( rvec, double, rvec, double, rvec );
void rvec_Sum( rvec, rvec, rvec );

double rvec_Dot( rvec, rvec );
void rvec_iMultiply( rvec, ivec, rvec );
void rvec_Cross( rvec, rvec, rvec );
double rvec_Norm_Sqr( rvec );
double rvec_Norm( rvec );
void rvec_MakeZero( rvec );
void rvec_Random( rvec );

void rtensor_MakeZero( rtensor );
void rtensor_MatVec( rvec, rtensor, rvec );

void ivec_MakeZero( ivec );
void ivec_Copy( ivec, ivec );
void ivec_Scale( ivec, double, ivec );
void ivec_Sum( ivec, ivec, ivec );

#endif
