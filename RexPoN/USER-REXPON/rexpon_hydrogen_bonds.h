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

#ifndef __HBONDS_H_
#define __HBONDS_H_

#include "rexpon_types.h"

void Hydrogen_Bonds( rexpon_system*, control_params*, simulation_data*,
                     storage*, rexpon_list**, output_controls* );

void Calculate_LonePair_Vecs( rvec, double, rvec, double, rvec*, rvec*,
                              rvec*, rvec* );
void Solve_Linear_Guassian(rvec, rvec, rvec, rvec, rvec* );
void Calculate_dCos_Alpha(rvec, rvec, int, rvec, double, rvec, double, rvec,
                          rvec, double, rvec*, rvec*);

#endif
