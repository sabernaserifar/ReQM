/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Saber Naserifar, Caltech, naseri@caltech.edu
------------------------------------------------------------------------- */

#include "math.h"
#include "math_const.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "pair_coul_pqeqgauss.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "memory.h"
#include "error.h"
#include "group.h"
#include "update.h"

using namespace LAMMPS_NS;
using namespace MathConst;

#define SQR(x) ((x)*(x))
#define CUBE(x) ((x)*(x)*(x))

/* ---------------------------------------------------------------------- */

PairCoulPqeqgauss::PairCoulPqeqgauss(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  writedata = 1;
  ghostneigh = 1;
}

/* ---------------------------------------------------------------------- */

PairCoulPqeqgauss::~PairCoulPqeqgauss()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(chi);
    memory->destroy(idem);
    memory->destroy(rcore);
    memory->destroy(polar);
    memory->destroy(qcore);
    memory->destroy(rshell);
    memory->destroy(kstring2);
    memory->destroy(kstring4);
    memory->destroy(specialc);
    memory->destroy(alphass);
    memory->destroy(alphasc);
    memory->destroy(alphacc);
    memory->destroy(swab);
    memory->destroy(cutghost);
  }
}

/* ---------------------------------------------------------------------- */

void PairCoulPqeqgauss::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum;
  int *ilist,*jlist,*numneigh,**firstneigh;
  int itype,jtype,polari,polarj;
  double ecoul,fpair,rsq,r,r2,factor_coul;
  double alpha,chii,idemi,kstring2i,kstring4i,kstring4j;
  double delx,dely,delz;
  double xi,yi,zi,rsxi,rsyi,rszi;
  double xj,yj,zj,rsxj,rsyj,rszj;
  double qi,qj,q1,q2,qcorei,qcorej,qshelli,qshellj;
  double shell_eng, coulombr, dcoulombr;

  ecoul = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  double **rsx = atom->rsx;
  int *type = atom->type;
  int *tag = atom->tag;
  int flag;
  double SMALL = 0.0001;
  int nlocal = atom->nlocal;

  double *special_coul = force->special_coul;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;
  double qe2f = force->qe2f;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];

    qi = q[i];
    xi = x[i][0];
    yi = x[i][1];
    zi = x[i][2];
    itype = type[i];
    polari = polar[itype];
    kstring4i = kstring4[itype];

    chii = qe2f*chi[itype];
    idemi = qe2f*idem[itype];

    jlist = firstneigh[i];
    jnum = numneigh[i];

    // self energy, only on i atom
    if (polari) {
      kstring2i = kstring2[itype];
      qcorei = qcore[itype];
      qshelli = -qcorei;
      rsxi = rsx[i][0];
      rsyi = rsx[i][1];
      rszi = rsx[i][2];
      r2 = rsxi*rsxi + rsyi*rsyi + rszi*rszi;
      r = sqrt(r2);
      shell_eng = 0.50 * kstring2i * r2 + kstring4i * r2 * r2;
    } else {
      shell_eng = 0.0;
      qcorei = 0.0;
    }

    if (evflag) { 
      ecoul = chii * qi + 0.5 * idemi * qi * qi + shell_eng;
      // VERY DANGEROUS change see below
      //if ( kstring4i < 0 ) ecoul = 0.0;
      ev_tally(i,i,nlocal,0,0.0,ecoul,0.0,0.0,0.0,0.0);
    }
 
    q1 = qcorei + qi;

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      jtype = type[j];
      kstring4j = kstring4[jtype];
      factor_coul = specialc[itype][sbmask(j)];
      // VERY DANGEROUS change
      // In REQM, we need PQEq charge based on the interactions in the 
      // entire system but we should exclude the PQEq force and energy interactions
      // btw the atoms in the QM region. I use the kstring4 parameter to identify the 
      // these interactions. Atoms in the QM region has kstring4=-1 otherwise kstring4=1
      // Therefore, for 2 atoms in QM region kstring4i+kstring4j<0 and we set 
      // factor_coul = 0  to exclude them 
      //if ( (kstring4i+kstring4j) < 0 ) {
          //factor_coul = 0.0 ;
      //}else{
          //factor_coul = 1.0 ; 
      //}

      xj = x[j][0];
      yj = x[j][1];
      zj = x[j][2];

      delx = xi - xj;
      dely = yi - yj;
      delz = zi - zj;
      r2 = delx*delx + dely*dely + delz*delz;

      flag = 0;
      if (r2 <= SQR(swb)) {
        if (j < nlocal) flag = 1;
        else if (tag[i] < tag[j]) flag = 1;
        else if (tag[i] == tag[j]) {
          if (-delz > SMALL) flag = 1;
          else if (fabs(delz) < SMALL) {
            if (-dely > SMALL) flag = 1;
            else if (fabs(dely) < SMALL && -delx > SMALL)
            flag = 1;
          }
        }
      }
      
      if (flag) {
        // get the pqeq parameters
        qj = q[j];
        polarj = polar[jtype];

        if (polarj) {
        qcorej = qcore[jtype];
        qshellj = -qcore[jtype];
        rsxj = rsx[j][0];
        rsyj = rsx[j][1];
        rszj = rsx[j][2];
        } else qcorej = 0.0;
        
        q2 = qcorej + qj;

        // core - core
        alpha = alphacc[itype][jtype];
        r = sqrt(r2);
        coulomb(r, alpha , &coulombr, &dcoulombr);
        fpair = - dcoulombr * q1 * q2 / r;
        if (evflag) {
          ecoul = factor_coul * coulombr * q1 * q2;
          ev_tally(i,j,nlocal,newton_pair,0.0,ecoul,fpair,delx,dely,delz);
        }
        f[i][0] += factor_coul*delx*fpair;
        f[i][1] += factor_coul*dely*fpair;
        f[i][2] += factor_coul*delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= factor_coul*delx*fpair;
          f[j][1] -= factor_coul*dely*fpair;
          f[j][2] -= factor_coul*delz*fpair;
        }

        // shell - shell
        if (polari and polarj) {
          alpha = alphass[itype][jtype];
          delx = rsxi - rsxj + delx;
          dely = rsyi - rsyj + dely;
          delz = rszi - rszj + delz;
          r2 = delx*delx + dely*dely + delz*delz;
          r = sqrt(r2);
          coulomb(r, alpha , &coulombr, &dcoulombr);
          fpair = - dcoulombr * qshelli * qshellj / r;
          if (evflag) {
            ecoul = factor_coul * coulombr * qshelli * qshellj;
            ev_tally(i,j,nlocal,newton_pair,0.0,ecoul,fpair,delx,dely,delz);
          }
          f[i][0] += factor_coul*delx*fpair;
          f[i][1] += factor_coul*dely*fpair;
          f[i][2] += factor_coul*delz*fpair;
          if (newton_pair || j < nlocal) {
            f[j][0] -= factor_coul*delx*fpair;
            f[j][1] -= factor_coul*dely*fpair;
            f[j][2] -= factor_coul*delz*fpair;
          }
        }

        // shell i - core j
        if (polari) {
          alpha = alphasc[itype][jtype];
          delx = (rsxi + xi) - xj;
          dely = (rsyi + yi) - yj;
          delz = (rszi + zi) - zj;
          r2 = delx*delx + dely*dely + delz*delz;
          r = sqrt(r2);
          coulomb(r, alpha , &coulombr, &dcoulombr);
          fpair = - dcoulombr * qshelli * q2 / r;
          if (evflag) {
            ecoul = factor_coul * coulombr * qshelli * q2;
            ev_tally(i,j,nlocal,newton_pair,0.0,ecoul,fpair,delx,dely,delz);
          }
          f[i][0] += factor_coul*delx*fpair;
          f[i][1] += factor_coul*dely*fpair;
          f[i][2] += factor_coul*delz*fpair;
          if (newton_pair || j < nlocal) {
            f[j][0] -= factor_coul*delx*fpair;
            f[j][1] -= factor_coul*dely*fpair;
            f[j][2] -= factor_coul*delz*fpair;
          }
        }

        // core i - shell j
        if (polarj) {
          alpha = alphasc[itype][jtype];
          delx = xi - (rsxj + xj);
          dely = yi - (rsyj + yj);
          delz = zi - (rszj + zj);
          r2 = delx*delx + dely*dely + delz*delz;
          r = sqrt(r2);
          coulomb(r, alpha , &coulombr, &dcoulombr);
          fpair = - dcoulombr * q1 * qshellj / r;
          if (evflag) {
            ecoul = factor_coul * coulombr * q1 * qshellj;
            ev_tally(i,j,nlocal,newton_pair,0.0,ecoul,fpair,delx,dely,delz);
          }
          f[i][0] += factor_coul*delx*fpair;
          f[i][1] += factor_coul*dely*fpair;
          f[i][2] += factor_coul*delz*fpair;
          if (newton_pair || j < nlocal) {
            f[j][0] -= factor_coul*delx*fpair;
            f[j][1] -= factor_coul*dely*fpair;
            f[j][2] -= factor_coul*delz*fpair;
          }
        }
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();

}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairCoulPqeqgauss::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(chi,n+1,"pair:chi");
  memory->create(idem,n+1,"pair:idem");
  memory->create(rcore,n+1,"pair:rcore");
  memory->create(polar,n+1,"pair:polar");
  memory->create(qcore,n+1,"pair:qcore");
  memory->create(rshell,n+1,"pair:rshell");
  memory->create(kstring2,n+1,"pair:kstring2");
  memory->create(kstring4,n+1,"pair:kstring4");
  memory->create(specialc,n+1,3,"pair:specialc");
  memory->create(alphass,n+1,n+1,"pair:alphass");
  memory->create(alphasc,n+1,n+1,"pair:alphasc");
  memory->create(alphacc,n+1,n+1,"pair:alphacc");
  memory->create(swab,2,"pair:cutoutin");
  memory->create(cutghost,n+1,n+1,"pair:cutghost");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairCoulPqeqgauss::settings(int narg, char **arg)
{
  if (narg != 2)
    error->all(FLERR,"Illegal pair_style command");

  swa = force->numeric(FLERR,arg[0]);
  swb = force->numeric(FLERR,arg[1]);
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairCoulPqeqgauss::coeff(int narg, char **arg)
{
  if (narg < 10)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);


  /* values for offdiagonal coefficients are ignored since the PQEq 
     formalism is diagnonal in the atom types*/
  if ((ilo != jlo) or (ihi != jhi)) {
    error->warning(FLERR,"Warning: when reading coul/pqeq pair coefficients, nondiagonal values are ignored");
  }

  double chi_one = force->numeric(FLERR,arg[2]);
  double idem_one = force->numeric(FLERR,arg[3]);
  double rcore_one = force->numeric(FLERR,arg[4]);
  int polar_one = force->inumeric(FLERR,arg[5]);
  double qcore_one = force->numeric(FLERR,arg[6]);
  double rshell_one = force->numeric(FLERR,arg[7]);  
  double kstring2_one = force->numeric(FLERR,arg[8]);
  double kstring4_one = force->numeric(FLERR,arg[9]);
  double *special_coul = force->special_coul;
  double special12_one = special_coul[0];
  double special13_one = special_coul[1];
  double special14_one = special_coul[2];

  if(narg>10) {
	  special12_one = force->numeric(FLERR,arg[10]);
	  if(narg>11) {
		  special13_one = force->numeric(FLERR,arg[11]);
		  if(narg>12) {
			  special14_one = force->numeric(FLERR,arg[12]);
		  }
	  }
  }

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = jlo; j <= jhi; j++) {
      if (i == j) {
        chi[i] = chi_one;
        idem[i] = idem_one;
        rcore[i] = rcore_one;
        polar[i] = polar_one;
        qcore[i] = qcore_one;
        rshell[i] = rshell_one;
        kstring2[i] = kstring2_one;
        kstring4[i] = kstring4_one;
		specialc[i][0] = special12_one;
		specialc[i][1] = special13_one;
		specialc[i][2] = special14_one;
        count++;
      }
      setflag[i][j] = 1;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/*------------------------------------------------------------------------- */

void PairCoulPqeqgauss::init_style()
{
  if (!atom->q_flag)
    error->all(FLERR,"Pair style coul/pqeqgauss requires atom attribute q");

  //int irequest = neighbor->request(this);
  
  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->newton = 2;
  neighbor->requests[irequest]->ghost = 1;

  if (swa >= swb )
    error->all(FLERR,"Pair inner cutoff >= Pair outer cutoff");

  swa2 = swa * swa;
  swb2 = swb * swb;
  swb3 = swb2 * swb;
  swa3 = swa2 * swa;

  init_taper();      
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
   this function gets called with j>=i
------------------------------------------------------------------------- */

double PairCoulPqeqgauss::init_one(int i, int j)
{
  double lambda = 0.462770;
  //double lambda = 1.00;

  double alpha_ci = lambda * 0.5 / rcore[i] / rcore[i];
  double alpha_cj = lambda * 0.5 / rcore[j] / rcore[j];

  alphacc[i][j] = sqrt(alpha_ci*alpha_cj/(alpha_ci+alpha_cj));
  alphacc[j][i] = alphacc[i][j];

  if (polar[i]) {
    double alpha_si = lambda * 0.5 / rshell[i] / rshell[i];
    alphasc[i][j] = sqrt(alpha_si*alpha_cj/(alpha_si+alpha_cj));   
    if  (polar[j]) {
      double alpha_sj = lambda * 0.5 / rshell[j] / rshell[j];
      alphasc[j][i] = sqrt(alpha_sj*alpha_ci/(alpha_sj+alpha_ci));   
      alphass[i][j] = sqrt(alpha_si*alpha_sj/(alpha_si+alpha_sj));
      alphass[j][i] = alphass[i][j];
    }
  } else {
    if  (polar[j]) {
      double alpha_sj = lambda * 0.5 / rshell[j] / rshell[j];
      alphasc[j][i] = sqrt(alpha_ci*alpha_sj/(alpha_ci+alpha_sj));
    }
  }

  cutghost[i][j] = cutghost[j][i] = swb;

  return swb;
}

/* ----------------------------------------------------------------------
  proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairCoulPqeqgauss::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    fwrite(&chi[i],sizeof(double),1,fp);
    fwrite(&idem[i],sizeof(double),1,fp);
    fwrite(&rcore[i],sizeof(double),1,fp);
    fwrite(&polar[i],sizeof(int),1,fp);
    fwrite(&qcore[i],sizeof(double),1,fp);
    fwrite(&rshell[i],sizeof(double),1,fp);
    fwrite(&kstring2[i],sizeof(double),1,fp);
    fwrite(&kstring4[i],sizeof(double),1,fp);
    fwrite(&specialc[i][0],sizeof(double),1,fp);
    fwrite(&specialc[i][1],sizeof(double),1,fp);
    fwrite(&specialc[i][2],sizeof(double),1,fp);
}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairCoulPqeqgauss::read_restart(FILE *fp)
{
  read_restart_settings(fp);

  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++) {
	if (me == 0) {  
      fread(&chi[i],sizeof(double),1,fp);
      fread(&idem[i],sizeof(double),1,fp);
      fread(&rcore[i],sizeof(double),1,fp);
      fread(&polar[i],sizeof(int),1,fp);
      fread(&qcore[i],sizeof(double),1,fp);
      fread(&rshell[i],sizeof(double),1,fp);
      fread(&kstring2[i],sizeof(double),1,fp);
      fread(&kstring4[i],sizeof(double),1,fp);
      fread(&specialc[i][0],sizeof(double),1,fp);
      fread(&specialc[i][1],sizeof(double),1,fp);
      fread(&specialc[i][2],sizeof(double),1,fp);
	}
	MPI_Bcast(&chi[i],1,MPI_DOUBLE,0,world);
	MPI_Bcast(&idem[i],1,MPI_DOUBLE,0,world);
	MPI_Bcast(&rcore[i],1,MPI_DOUBLE,0,world);
	MPI_Bcast(&polar[i],1,MPI_INT,0,world);
	MPI_Bcast(&qcore[i],1,MPI_DOUBLE,0,world);
	MPI_Bcast(&rshell[i],1,MPI_DOUBLE,0,world);
	MPI_Bcast(&kstring2[i],1,MPI_DOUBLE,0,world);
	MPI_Bcast(&kstring4[i],1,MPI_DOUBLE,0,world);
	MPI_Bcast(&specialc[i],3,MPI_DOUBLE,0,world);
  }
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairCoulPqeqgauss::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp,"%d %g %g %g %d %g %g %g %g %g %g %g\n",
       i,chi[i],idem[i],rcore[i],polar[i],qcore[i],rshell[i],kstring2[i],kstring4[i],specialc[i][0],specialc[i][1],specialc[i][2]);
}

/* ----------------------------------------------------------------------
  proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairCoulPqeqgauss::write_restart_settings(FILE *fp)
{
  fwrite(&swa,sizeof(double),1,fp);
  fwrite(&swb,sizeof(double),1,fp);
}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairCoulPqeqgauss::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    fread(&swa,sizeof(double),1,fp);
    fread(&swb,sizeof(double),1,fp);
  }
  MPI_Bcast(&swa,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&swb,1,MPI_DOUBLE,0,world);
}

/* -----------------------------------------------------------------------
   Intializes the taper, as in fix_qeq_reax
 -------------------------------------------------------------------------- */

void PairCoulPqeqgauss::init_taper()
{
  double d7, swa2, swa3, swb2, swb3;

  if (fabs(swa) > 0.01 && comm->me == 0)
    error->warning(FLERR,"PairCoulPqeqgauss has non-zero lower Taper radius cutoff");
  if (swb < 0)
    error->all(FLERR, "PairCoulPqeqgauss has negative upper Taper radius cutoff");
  else if (swa < 0 && comm->me == 0)
    error->warning(FLERR,"PairCoulPqeqgauss has negative low Taper radius cutoff");

  d7 = pow( swb - swa, 7 );
  swa2 = SQR( swa );
  swa3 = CUBE( swa );
  swb2 = SQR( swb );
  swb3 = CUBE( swb );

  Tap[7] =  20.0 / d7;
  Tap[6] = -70.0 * (swa + swb) / d7;
  Tap[5] =  84.0 * (swa2 + 3.0*swa*swb + swb2) / d7;
  Tap[4] = -35.0 * (swa3 + 9.0*swa2*swb + 9.0*swa*swb2 + swb3 ) / d7;
  Tap[3] = 140.0 * (swa3*swb + 3.0*swa2*swb2 + swa*swb3 ) / d7;
  Tap[2] =-210.0 * (swa3*swb2 + swa2*swb3) / d7;
  Tap[1] = 140.0 * swa3 * swb3 / d7;
  Tap[0] = (-35.0*swa3*swb2*swb2 + 21.0*swa2*swb3*swb2 +
            7.0*swa*swb3*swb3 + swb3*swb3*swb ) / d7;

  //taper derivative
  dTap[7] = 0.0;
  dTap[6] = 7.0*Tap[7];
  dTap[5] = 6.0*Tap[6];
  dTap[4] = 5.0*Tap[5];
  dTap[3] = 4.0*Tap[4];
  dTap[2] = 3.0*Tap[3];
  dTap[1] = 2.0*Tap[2];
  dTap[0] = Tap[1];
}

/* -----------------------------------------------------------------------
   Our modified Coulomb (r has to be r<swb)
-------------------------------------------------------------------------- */

void PairCoulPqeqgauss::coulomb(double r, double alpha, double *coulombr, double *dcoulombr)
{
  int n;
  double taper,dtaper,screening,dscreening,coul,dcoul,r2;
  double qqrd2e = force->qqrd2e;

  r2 = r * r;
  // taper to smoothy go to zero between swa and swb
  if (r > swa) {
    taper  =  Tap[7];
    dtaper = dTap[7];
    for(int n=6; n>=0; n--){
      taper  = taper * r + Tap[n];
      dtaper = dtaper * r + dTap[n];
    }
  } else {
    taper = 1.0;
    dtaper = 0.0;
  }

  // for shells, force to zero when the distance is outside of cutoff 
  if ( r > swb) {
    taper = 0.0;
    dtaper = 0.0;
  }

  coul = qqrd2e / r;
  dcoul = - qqrd2e / r2;

  screening = erf(alpha * r);
  dscreening = 2.0 * alpha * exp(- alpha * alpha * r2) / MY_PIS;
  *coulombr = coul * taper * screening;
  *dcoulombr = dcoul * taper * screening + coul * dtaper * screening +
          coul * taper * dscreening;
}

/* ---------------------------------------------------------------------- */

void *PairCoulPqeqgauss::extract(const char *str, int &dim)
{
  dim = 2;
  if (strcmp(str,"alphass") == 0) return (void *) alphass;
  if (strcmp(str,"alphasc") == 0) return (void *) alphasc;
  if (strcmp(str,"alphacc") == 0) return (void *) alphacc;
  if (strcmp(str,"specialc") == 0) return (void *) specialc;

  dim = 1;
  if (strcmp(str,"chi") == 0) return (void *) chi;
  if (strcmp(str,"idem") == 0) return (void *) idem;
  if (strcmp(str,"rcore") == 0) return (void *) rcore;
  if (strcmp(str,"rshell") == 0) return (void *) rshell;
  if (strcmp(str,"qcore") == 0) return (void *) qcore;
  if (strcmp(str,"kstring2") == 0) return (void *) kstring2;
  if (strcmp(str,"kstring4") == 0) return (void *) kstring4;
  if (strcmp(str,"polar") == 0) return (void *) polar;
  if (strcmp(str,"Tap") == 0) return (void *) Tap;
  if (strcmp(str,"dTap") == 0) return (void *) dTap;

  dim = 0;
  if (strcmp(str,"swa") == 0) return (void *) &swa;
  if (strcmp(str,"swb") == 0) return (void *) &swb;

  return NULL;
}
