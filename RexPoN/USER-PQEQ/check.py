import os, sys, glob

files = glob.glob('*')


for f in files:
   print ("######################################################", f)
   os.system('diff %s ../%s'%(f, f))
