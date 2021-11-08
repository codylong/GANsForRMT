import os
import cPickle as pickle

files = [d for d in os.listdir('data') if d[0] == 'h' and 'evals' in d]
print 'num files', len(files)
assert_mismatch = []

def info_from_file(file):
    h11 = int(file.split("_")[1])
    s = file.split("_")[2]
    poly = int(s[s.index('y')+1:])
    f = open("data/"+file,'r')
    evals = pickle.load(f)
    f.close()
    to_ret = [(h11,poly),evals]
    try:
        assert len(evals) == h11
    except AssertionError:
        assert_mismatch.append(to_ret[0])
    return to_ret

raw_data = [info_from_file(file) for file in files]
print set([len(r[1]) - r[0][0] for r in raw_data])
raw_data = [r for r in raw_data if len(r[1]) - r[0][0] == 0]
print "num files after eval mismatch subtraction", len(raw_data)
print assert_mismatch

from mpi4py import MPI
comm = MPI.COMM_WORLD
size, rank = int(comm.Get_size()), int(comm.Get_rank())

print 'rank', 10*(rank+1)
h11 = 10*(rank+1)

import datetime
from scipy.stats import wasserstein_distance
raw = [r for r in raw_data if r[0][0] == 10]
print len(raw)
wasser_dists = []
for i1 in range(len(raw)):
    if i1 % 100 == 0: print i1, datetime.datetime.now()
    for i2 in range(i1+1,len(raw)):
        wasser_dists.append(wasserstein_distance(raw[i1][1],raw[i2][1]))

f = open("wasser_dists/wasser_dist_h11_" + str(h11) + ".pickle",'w')
pickle.dump(wasser_dists,f)
f.close()
