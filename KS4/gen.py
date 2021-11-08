import cPickle as pickle
import subprocess
import os.path
import time
from scipy import special
import numpy as np
import math
#from Notebooks.sigthresh import *

#outdir = 'inout2'

def writeScript(h11,ns,nn):
    out = "#!/bin/bash"
    out += "\n#SBATCH --job-name=test"
    out += "\n#SBATCH --output=test.out"
    out += "\n#SBATCH --error=test.err"
    out += "\n#SBATCH --exclusive"
    #old name for discovery node
    #out += "\n#SBATCH --partition=ser-par-10g-5"
    out += "\n#SBATCH --partition=fullnode"
    out += "\n#SBATCH -N 1"
    out += "\n#SBATCH --workdir=/home/codylong/KS4/workdir"
    out += "\ncd /home/codylong/KS4"
    out += "\n/shared/apps/sage-7.4/sage general_extra_for_rand_sugra2.sage " + str(h11) + ' ' + str(ns) + ' ' + str(nn)
            
    f = open("/home/codylong/KS4/scripts/test.job",'w')
    f.write(out)
    f.close()
    output=subprocess.Popen("sbatch /home/codylong/AxionReheating/scripts/test.job",shell=True,stdout=subprocess.PIPE).communicate()[0]
    return output
for zzz in range(10):
    writeScript(25,zzz*10,100)
