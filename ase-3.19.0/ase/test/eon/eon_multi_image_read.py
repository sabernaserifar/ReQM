# flake8: noqa
"""Check that reading multi image .con files is consistent."""

import tempfile
import os
import shutil

from numpy import array
import ase
import ase.io


# Error tolerance.
TOL = 1e-6

# A correct .con file.
CON_FILE = """\
0	Random Number Seed
0	Time
8.123222	5.744000	9.747867
90.000000	90.000000	90.000000
0 0
0 0 0
1
17
51.996100
Cr
Coordinates of Component 1
   1.01540277999999962    0.71799999999999997    1.01540277999999984 1    0
   3.04620834000000063    2.15399999999999991    1.01540277999999984 1    1
   3.04620834000000063    0.71799999999999997    3.04620834000000196 1    2
   1.01540277999999962    2.15399999999999991    3.04620834000000196 1    3
   1.01540277999999962    3.58999999999999986    1.01540277999999984 1    4
   3.04620834000000063    5.02599999999999980    1.01540277999999984 1    5
   3.04620834000000063    3.58999999999999986    3.04620834000000196 1    6
   1.01540277999999962    5.02599999999999980    3.04620834000000196 1    7
   5.07701389999999986    0.71799999999999997    1.01540277999999984 1    8
   7.10781945999998488    2.15399999999999991    1.01540277999999984 1    9
   7.10781945999998488    0.71799999999999997    3.04620834000000196 1   10
   5.07701389999999986    2.15399999999999991    3.04620834000000196 1   11
   5.07701389999999986    3.58999999999999986    1.01540277999999984 1   12
   7.10781945999998488    5.02599999999999980    1.01540277999999984 1   13
   7.10781945999998488    3.58999999999999986    3.04620834000000196 1   14
   5.07701389999999986    5.02599999999999980    3.04620834000000196 1   15
   3.04618285858587523    2.15398224021542450    4.60622193000079427 0   16
0	Random Number Seed
0	Time
8.123222	5.744000	9.747867
90.000000	90.000000	90.000000
0 0
0 0 0
1
17
51.996100
Cr
Coordinates of Component 1
   1.01540277999999962    0.71799999999999997    1.01540277999999984 1    0
   3.04620834000000063    2.15399999999999991    1.01540277999999984 1    1
   3.04620834000000063    0.71799999999999997    3.04620834000000196 1    2
   1.01540277999999962    2.15399999999999991    3.04620834000000196 1    3
   1.01540277999999962    3.58999999999999986    1.01540277999999984 1    4
   3.04620834000000063    5.02599999999999980    1.01540277999999984 1    5
   3.04620834000000063    3.58999999999999986    3.04620834000000196 1    6
   1.01540277999999962    5.02599999999999980    3.04620834000000196 1    7
   5.07701389999999986    0.71799999999999997    1.01540277999999984 1    8
   7.10781945999998488    2.15399999999999991    1.01540277999999984 1    9
   7.10781945999998488    0.71799999999999997    3.04620834000000196 1   10
   5.07701389999999986    2.15399999999999991    3.04620834000000196 1   11
   5.07701389999999986    3.58999999999999986    1.01540277999999984 1   12
   7.10781945999998488    5.02599999999999980    1.01540277999999984 1   13
   7.10781945999998488    3.58999999999999986    3.04620834000000196 1   14
   5.07701389999999986    5.02599999999999980    3.04620834000000196 1   15
   3.36369427985916092    2.20887986058760699    4.61557342394151693 0   16
0	Random Number Seed
0	Time
8.123222	5.744000	9.747867
90.000000	90.000000	90.000000
0 0
0 0 0
1
17
51.996100
Cr
Coordinates of Component 1
   1.01540277999999962    0.71799999999999997    1.01540277999999984 1    0
   3.04620834000000063    2.15399999999999991    1.01540277999999984 1    1
   3.04620834000000063    0.71799999999999997    3.04620834000000196 1    2
   1.01540277999999962    2.15399999999999991    3.04620834000000196 1    3
   1.01540277999999962    3.58999999999999986    1.01540277999999984 1    4
   3.04620834000000063    5.02599999999999980    1.01540277999999984 1    5
   3.04620834000000063    3.58999999999999986    3.04620834000000196 1    6
   1.01540277999999962    5.02599999999999980    3.04620834000000196 1    7
   5.07701389999999986    0.71799999999999997    1.01540277999999984 1    8
   7.10781945999998488    2.15399999999999991    1.01540277999999984 1    9
   7.10781945999998488    0.71799999999999997    3.04620834000000196 1   10
   5.07701389999999986    2.15399999999999991    3.04620834000000196 1   11
   5.07701389999999986    3.58999999999999986    1.01540277999999984 1   12
   7.10781945999998488    5.02599999999999980    1.01540277999999984 1   13
   7.10781945999998488    3.58999999999999986    3.04620834000000196 1   14
   5.07701389999999986    5.02599999999999980    3.04620834000000196 1   15
   3.62116697589668135    2.40183843231018113    4.63674682635805180 0   16
0	Random Number Seed
0	Time
8.123222	5.744000	9.747867
90.000000	90.000000	90.000000
0 0
0 0 0
1
17
51.996100
Cr
Coordinates of Component 1
   1.01540277999999962    0.71799999999999997    1.01540277999999984 1    0
   3.04620834000000063    2.15399999999999991    1.01540277999999984 1    1
   3.04620834000000063    0.71799999999999997    3.04620834000000196 1    2
   1.01540277999999962    2.15399999999999991    3.04620834000000196 1    3
   1.01540277999999962    3.58999999999999986    1.01540277999999984 1    4
   3.04620834000000063    5.02599999999999980    1.01540277999999984 1    5
   3.04620834000000063    3.58999999999999986    3.04620834000000196 1    6
   1.01540277999999962    5.02599999999999980    3.04620834000000196 1    7
   5.07701389999999986    0.71799999999999997    1.01540277999999984 1    8
   7.10781945999998488    2.15399999999999991    1.01540277999999984 1    9
   7.10781945999998488    0.71799999999999997    3.04620834000000196 1   10
   5.07701389999999986    2.15399999999999991    3.04620834000000196 1   11
   5.07701389999999986    3.58999999999999986    1.01540277999999984 1   12
   7.10781945999998488    5.02599999999999980    1.01540277999999984 1   13
   7.10781945999998488    3.58999999999999986    3.04620834000000196 1   14
   5.07701389999999986    5.02599999999999980    3.04620834000000196 1   15
   3.83933949582109157    2.63825709178043821    4.65669727965894875 0   16
0	Random Number Seed
0	Time
8.123222	5.744000	9.747867
90.000000	90.000000	90.000000
0 0
0 0 0
1
17
51.996100
Cr
Coordinates of Component 1
   1.01540277999999962    0.71799999999999997    1.01540277999999984 1    0
   3.04620834000000063    2.15399999999999991    1.01540277999999984 1    1
   3.04620834000000063    0.71799999999999997    3.04620834000000196 1    2
   1.01540277999999962    2.15399999999999991    3.04620834000000196 1    3
   1.01540277999999962    3.58999999999999986    1.01540277999999984 1    4
   3.04620834000000063    5.02599999999999980    1.01540277999999984 1    5
   3.04620834000000063    3.58999999999999986    3.04620834000000196 1    6
   1.01540277999999962    5.02599999999999980    3.04620834000000196 1    7
   5.07701389999999986    0.71799999999999997    1.01540277999999984 1    8
   7.10781945999998488    2.15399999999999991    1.01540277999999984 1    9
   7.10781945999998488    0.71799999999999997    3.04620834000000196 1   10
   5.07701389999999986    2.15399999999999991    3.04620834000000196 1   11
   5.07701389999999986    3.58999999999999986    1.01540277999999984 1   12
   7.10781945999998488    5.02599999999999980    1.01540277999999984 1   13
   7.10781945999998488    3.58999999999999986    3.04620834000000196 1   14
   5.07701389999999986    5.02599999999999980    3.04620834000000196 1   15
   4.06162234392075128    2.87141228075409316    4.66819033729618926 0   16
0	Random Number Seed
0	Time
8.123222	5.744000	9.747867
90.000000	90.000000	90.000000
0 0
0 0 0
1
17
51.996100
Cr
Coordinates of Component 1
   1.01540277999999962    0.71799999999999997    1.01540277999999984 1    0
   3.04620834000000063    2.15399999999999991    1.01540277999999984 1    1
   3.04620834000000063    0.71799999999999997    3.04620834000000196 1    2
   1.01540277999999962    2.15399999999999991    3.04620834000000196 1    3
   1.01540277999999962    3.58999999999999986    1.01540277999999984 1    4
   3.04620834000000063    5.02599999999999980    1.01540277999999984 1    5
   3.04620834000000063    3.58999999999999986    3.04620834000000196 1    6
   1.01540277999999962    5.02599999999999980    3.04620834000000196 1    7
   5.07701389999999986    0.71799999999999997    1.01540277999999984 1    8
   7.10781945999998488    2.15399999999999991    1.01540277999999984 1    9
   7.10781945999998488    0.71799999999999997    3.04620834000000196 1   10
   5.07701389999999986    2.15399999999999991    3.04620834000000196 1   11
   5.07701389999999986    3.58999999999999986    1.01540277999999984 1   12
   7.10781945999998488    5.02599999999999980    1.01540277999999984 1   13
   7.10781945999998488    3.58999999999999986    3.04620834000000196 1   14
   5.07701389999999986    5.02599999999999980    3.04620834000000196 1   15
   4.28380999862949441    3.10482201783771883    4.65660558580467221 0   16
0	Random Number Seed
0	Time
8.123222	5.744000	9.747867
90.000000	90.000000	90.000000
0 0
0 0 0
1
17
51.996100
Cr
Coordinates of Component 1
   1.01540277999999962    0.71799999999999997    1.01540277999999984 1    0
   3.04620834000000063    2.15399999999999991    1.01540277999999984 1    1
   3.04620834000000063    0.71799999999999997    3.04620834000000196 1    2
   1.01540277999999962    2.15399999999999991    3.04620834000000196 1    3
   1.01540277999999962    3.58999999999999986    1.01540277999999984 1    4
   3.04620834000000063    5.02599999999999980    1.01540277999999984 1    5
   3.04620834000000063    3.58999999999999986    3.04620834000000196 1    6
   1.01540277999999962    5.02599999999999980    3.04620834000000196 1    7
   5.07701389999999986    0.71799999999999997    1.01540277999999984 1    8
   7.10781945999998488    2.15399999999999991    1.01540277999999984 1    9
   7.10781945999998488    0.71799999999999997    3.04620834000000196 1   10
   5.07701389999999986    2.15399999999999991    3.04620834000000196 1   11
   5.07701389999999986    3.58999999999999986    1.01540277999999984 1   12
   7.10781945999998488    5.02599999999999980    1.01540277999999984 1   13
   7.10781945999998488    3.58999999999999986    3.04620834000000196 1   14
   5.07701389999999986    5.02599999999999980    3.04620834000000196 1   15
   4.50188452903429326    3.34154720502221236    4.63664894132718874 0   16
0	Random Number Seed
0	Time
8.123222	5.744000	9.747867
90.000000	90.000000	90.000000
0 0
0 0 0
1
17
51.996100
Cr
Coordinates of Component 1
   1.01540277999999962    0.71799999999999997    1.01540277999999984 1    0
   3.04620834000000063    2.15399999999999991    1.01540277999999984 1    1
   3.04620834000000063    0.71799999999999997    3.04620834000000196 1    2
   1.01540277999999962    2.15399999999999991    3.04620834000000196 1    3
   1.01540277999999962    3.58999999999999986    1.01540277999999984 1    4
   3.04620834000000063    5.02599999999999980    1.01540277999999984 1    5
   3.04620834000000063    3.58999999999999986    3.04620834000000196 1    6
   1.01540277999999962    5.02599999999999980    3.04620834000000196 1    7
   5.07701389999999986    0.71799999999999997    1.01540277999999984 1    8
   7.10781945999998488    2.15399999999999991    1.01540277999999984 1    9
   7.10781945999998488    0.71799999999999997    3.04620834000000196 1   10
   5.07701389999999986    2.15399999999999991    3.04620834000000196 1   11
   5.07701389999999986    3.58999999999999986    1.01540277999999984 1   12
   7.10781945999998488    5.02599999999999980    1.01540277999999984 1   13
   7.10781945999998488    3.58999999999999986    3.04620834000000196 1   14
   5.07701389999999986    5.02599999999999980    3.04620834000000196 1   15
   4.75928919917819293    3.53496190773495211    4.61566200013953409 0   16
0	Random Number Seed
0	Time
8.123222	5.744000	9.747867
90.000000	90.000000	90.000000
0 0
0 0 0
1
17
51.996100
Cr
Coordinates of Component 1
   1.01540277999999962    0.71799999999999997    1.01540277999999984 1    0
   3.04620834000000063    2.15399999999999991    1.01540277999999984 1    1
   3.04620834000000063    0.71799999999999997    3.04620834000000196 1    2
   1.01540277999999962    2.15399999999999991    3.04620834000000196 1    3
   1.01540277999999962    3.58999999999999986    1.01540277999999984 1    4
   3.04620834000000063    5.02599999999999980    1.01540277999999984 1    5
   3.04620834000000063    3.58999999999999986    3.04620834000000196 1    6
   1.01540277999999962    5.02599999999999980    3.04620834000000196 1    7
   5.07701389999999986    0.71799999999999997    1.01540277999999984 1    8
   7.10781945999998488    2.15399999999999991    1.01540277999999984 1    9
   7.10781945999998488    0.71799999999999997    3.04620834000000196 1   10
   5.07701389999999986    2.15399999999999991    3.04620834000000196 1   11
   5.07701389999999986    3.58999999999999986    1.01540277999999984 1   12
   7.10781945999998488    5.02599999999999980    1.01540277999999984 1   13
   7.10781945999998488    3.58999999999999986    3.04620834000000196 1   14
   5.07701389999999986    5.02599999999999980    3.04620834000000196 1   15
   5.07701160164306042    3.58998956883621734    4.60626159988447537 0   16
"""

# The corresponding data for the second to last image as an ASE Atoms object.
data = ase.Atoms('Cr17',cell = array([[8.123222, 0, 0],
                                      [0, 5.744000, 0],
                                      [0, 0, 9.747867]]),
                        positions = array([ [1.01540277999999962,    0.71799999999999997,    1.01540277999999984],
                                            [3.04620834000000063,    2.15399999999999991,    1.01540277999999984],
                                            [3.04620834000000063,    0.71799999999999997,    3.04620834000000196],
                                            [1.01540277999999962,    2.15399999999999991,    3.04620834000000196],
                                            [1.01540277999999962,    3.58999999999999986,    1.01540277999999984],
                                            [3.04620834000000063,    5.02599999999999980,    1.01540277999999984],
                                            [3.04620834000000063,    3.58999999999999986,    3.04620834000000196],
                                            [1.01540277999999962,    5.02599999999999980,    3.04620834000000196],
                                            [5.07701389999999986,    0.71799999999999997,    1.01540277999999984],
                                            [7.10781945999998488,    2.15399999999999991,    1.01540277999999984],
                                            [7.10781945999998488,    0.71799999999999997,    3.04620834000000196],
                                            [5.07701389999999986,    2.15399999999999991,    3.04620834000000196],
                                            [5.07701389999999986,    3.58999999999999986,    1.01540277999999984],
                                            [7.10781945999998488,    5.02599999999999980,    1.01540277999999984],
                                            [7.10781945999998488,    3.58999999999999986,    3.04620834000000196],
                                            [5.07701389999999986,    5.02599999999999980,    3.04620834000000196],
                                            [4.75928919917819293,    3.53496190773495211,    4.61566200013953409]]),
                        pbc=(True, True, True))





tempdir = tempfile.mkdtemp()
try:
    # First, write a correct .con file and try to read it.
    con_file = os.path.join(tempdir, 'neb.con')
    with open(con_file, 'w') as f:
        f.write(CON_FILE)
    images = ase.io.read(con_file, format='eon', index =':')
    box = images[-2]
    # Check cell vectors.
    assert (abs(box.cell - data.cell)).sum() < TOL  # read: cell vector check
    # Check atom positions.
    # read: position check
    assert (abs(box.positions - data.positions)).sum() < TOL

    # Now that we know that reading a .con file works, we will write
    # one and read it back in.
    out_file = os.path.join(tempdir, 'out.con')
    ase.io.write(out_file, data, format='eon')
    data2 = ase.io.read(out_file, format='eon')
    # Check cell vectors.
    # write: cell vector check
    assert (abs(data2.cell - data.cell)).sum() < TOL
    # Check atom positions.
    # write: position check
    assert (abs(data2.positions - data.positions)).sum() < TOL
finally:
    shutil.rmtree(tempdir)
