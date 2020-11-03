import os
from ase import Atoms
#from mylammps import LAMMPS
from ase.calculators.lammpsrun import LAMMPS
from ase.constraints import FixAtoms, FixBondLengths
from ase.md import MDLogger, Langevin, npt, nvtberendsen
import ase.units as units
from ase.io.trajectory import Trajectory
from ase.io import write,read
from ase.io.vasp import read_vasp, write_vasp
from ase.calculators.vasp import Vasp
from myqmmm import deQMMM
from ase.optimize import BFGS, FIRE
from ase.visualize import view
from ase.constraints import Hookean
from ase.calculators.qmmm import SimpleQMMM
from ase.optimize import MDMin, FIRE, BFGS, BFGSLineSearch, LBFGS
import numpy, copy
#-------------- Set up inputs and regions---------------------#
tag = 'min'
atoms = read_vasp('CONTCAR_AuCO_01007_solvated_FFrelaxed.vasp')
molid = numpy.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39, 40, 40, 41, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 50, 50, 51, 51, 52, 52, 53, 53, 54, 54, 55, 55, 56, 56, 57, 57, 59, 59, 60, 60, 62, 62, 63, 63, 64, 64, 65, 65, 67, 67, 68, 68, 69, 69, 70, 70, 71, 71, 73, 73, 74, 74, 76, 76, 78, 78, 79, 79, 81, 81, 82, 82, 83, 83, 84, 84, 85, 85, 86, 86, 87, 87, 88, 88, 89, 89, 90, 90, 91, 91, 92, 92, 93, 93, 94, 94, 96, 96, 97, 97, 98, 98, 99, 99, 100, 100, 101, 101, 102, 102, 103, 103, 104, 104, 105, 105, 106, 106, 107, 107, 108, 108, 110, 110, 111, 111, 112, 112, 113, 113, 114, 114, 115, 115, 116, 116, 117, 117, 118, 118, 119, 119, 120, 120, 121, 121, 122, 122, 123, 123, 124, 124, 125, 125, 126, 126, 127, 127, 128, 128, 129, 129, 130, 130, 131, 131, 132, 132, 133, 133, 134, 134, 135, 135, 136, 136, 138, 138, 141, 141, 143, 143, 144, 144, 145, 145, 147, 147, 149, 149, 150, 150, 151, 151, 152, 152, 153, 153, 154, 154, 155, 155, 156, 156, 157, 157, 158, 158, 159, 159, 160, 160, 161, 161, 162, 162, 163, 163, 164, 164, 165, 165, 166, 166, 167, 167, 168, 168, 169, 169, 171, 171, 172, 172, 173, 173, 174, 174, 175, 175, 176, 176, 177, 177, 179, 179, 180, 180, 181, 181, 182, 182, 183, 183, 184, 184, 185, 185, 186, 186, 188, 188, 189, 189, 190, 190, 191, 191, 192, 192, 193, 193, 194, 194, 195, 195, 196, 196, 197, 197, 198, 198, 200, 200, 201, 201, 203, 203, 204, 204, 205, 205, 206, 206, 207, 207, 208, 208, 209, 209, 210, 210, 212, 212, 213, 213, 214, 214, 215, 215, 217, 217, 218, 218, 219, 219, 222, 222, 224, 224, 225, 225, 226, 226, 227, 227, 228, 228, 229, 229, 230, 230, 231, 231, 232, 232, 234, 234, 238, 238, 239, 239, 240, 240, 241, 241, 242, 242, 243, 243, 245, 245, 247, 247, 248, 248, 249, 249, 250, 250, 251, 251, 252, 252, 253, 253, 254, 254, 255, 255, 256, 256, 260, 260, 261, 261, 262, 262, 263, 263, 264, 264, 265, 265, 267, 267, 268, 268, 269, 269, 271, 271, 272, 272, 273, 273, 274, 274, 275, 275, 276, 276, 277, 277, 278, 278, 279, 279, 280, 280, 282, 282, 283, 283, 284, 284, 285, 285, 286, 286, 287, 287, 288, 288, 289, 289, 293, 293, 294, 294, 295, 295, 296, 296, 297, 297, 298, 298, 299, 299, 300, 300, 302, 302, 303, 303, 305, 305, 306, 306, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 62, 63, 64, 65, 67, 68, 69, 70, 71, 73, 74, 76, 78, 79, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 138, 141, 143, 144, 145, 147, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 171, 172, 173, 174, 175, 176, 177, 179, 180, 181, 182, 183, 184, 185, 186, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 200, 201, 203, 204, 205, 206, 207, 208, 209, 210, 212, 213, 214, 215, 217, 218, 219, 222, 224, 225, 226, 227, 228, 229, 230, 231, 232, 234, 238, 239, 240, 241, 242, 243, 245, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 260, 261, 262, 263, 264, 265, 267, 268, 269, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 282, 283, 284, 285, 286, 287, 288, 289, 293, 294, 295, 296, 297, 298, 299, 300, 302, 303, 305, 306, 308, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 308, 307])
atoms.arrays['mol_id'] = molid
#qmregion = [i for i in range()]
qmregion = [777, 783, 784, 797, 809, 811, 815, 818, 825, 826, 836, 837, 838, 839, 841, 842, 843]

qmwaters = [521, 6, 7, 534, 32, 33, 550, 64, 65, 560, 84, 85, 575, 114, 115, 587, 138, 139, 594, 152, 153, 596, 156, 157, 605, 174, 175, 608, 180, 181, 627, 218, 219, 645, 254, 255, 678, 320, 321, 703, 370, 371, 729, 422, 423, 737, 438, 439, 740, 444, 445, 753, 470, 471, 761, 486, 487, 762, 488, 489]
#qmregion += qmwaters 
fixedatoms = [i for i in range(0,842)]
brregion = []
qmbrregion = qmregion + brregion


for atom in atoms:
    if (atom.symbol=='C'):
       c_bond = atom.index
       fixedatoms.append(atom.index)

#-------------- Set up constrainsts on atoms---------------------#
atoms.constraints = FixAtoms(indices=fixedatoms)
#c = FixBondLengths([[777,c_bond]])
#atoms.constraints.append(c)

#--------------- Set up Vasp calcualtions---------------------#
os.environ['VASP_COMMAND'] = 'srun /central/groups/wag/programs/vasp.5.4.4/bin/vasp_gam' #! define VASP binary path
os.environ['VASP_PP_PATH'] = '/central/home/naseri/bin/potcar/' #! define VASP pseudopotential library path
qmcalc = Vasp(tmp_dir = './vasp_run',
              track_output=True,
              system = 'CONTCAR_AuCO_01007_solvated_FFrelaxed.vasp',
              npar = 4,
              istart = 0,
              icharg = 2,
              iniwav = 1,
              encut = 400.0,
              xc = 'PBE',
              voskown = 1,
              ivdw = 11,
              prec = 'normal',
              nelm = 200,
              ediff=1.0E-05,
              ismear = 1,
              sigma = 0.2,
              isym = 2,
              lreal = 'Auto',
              ialgo = 48,
              ediffg= -0.01,
              ibrion = 2,
              isif = 2,
              lwave = False,
              lcharg = False,
              nwrite = 2,
              kpts = (1,1,1),
              gamma= False
             )
#--------------- Set up RexPoN calcualtions---------------------#
#os.environ['ASE_LAMMPSRUN_COMMAND'] = "/central/slurm/install/current/bin/srun -n 8 /home/naseri/codes/src-REQM/lmp_intel"
os.environ['ASE_LAMMPSRUN_COMMAND'] = "/home/naseri/codes/src-REQM/lmp_intel"

files = ['/home/naseri/bin/reqm/ffield.RexPoN']
parameters = {"atom_style": "pqeq",
              "atom_modify": "map hash",
              "units": "real",
              "boundary": "p p p",
              "pair_style": "hybrid/overlay coul/pqeqgauss 0 12.0 rexpon NULL lgvdw yes coulomb_off yes checkqeq no",
              "pair_coeff": [' * * rexpon ffield.RexPoN Hw Ow Au C Ow ',
                        ' * * coul/pqeqgauss 0 0 0 0 0 0 0 0',
                        '1 1 coul/pqeqgauss 4.528000 17.984100 0.302857 1 1.000000 0.302857 2037.20060 0.0',
                        '2 2 coul/pqeqgauss 8.741000 13.364000 0.546120 1 1.000000 0.546120 814.04450 0.0',
                        '3 3 coul/pqeqgauss 4.894000  5.172000 1.618000 1 1.000000 1.618000 740.00000 0.000000',
                        '4 4 coul/pqeqgauss 5.50813   9.81186 0.75900 1 1.000000 0.75900 198.84054 0.000000',
                        '5 5 coul/pqeqgauss 8.741000 13.364000 0.546120 1 1.000000 0.546120 814.04450 0.000000'],
              "masses": ['1 1.00784', '2 15.99940', '3 196.96655', '4 12.01070', '5 15.99940'],
              "fix": ["pqeq all pqeq"],
              "neighbor": "2.5 bin"}

#mmcalc = LAMMPS(parameters=parameters, files=files, tmp_dir="./lammps_run")
mmcalc = LAMMPS(parameters=parameters, specorder=['Hw', 'Ow', 'Au', 'C', 'O'], files=files, tmp_dir="./lammps_run", keep_tmp_files=False)

#--------------- Set up REQM calcualtions---------------------#
atoms.calc = SimpleQMMM(qmbrregion,
                        qmcalc,
                        mmcalc,mmcalc)
#atoms.set_calculator(qmcalc)
#atoms.set_calculator(mmcalc)
#atoms.set_calculator(qmmmcalc)


#--------------- Set up Optimization ---------------------#
traj = Trajectory(tag + '.ase', 'w', atoms)
optimizer = BFGS(atoms)
optimizer.attach(traj.write, interval=1)
optimizer.run(fmax=0.05, steps=35)


