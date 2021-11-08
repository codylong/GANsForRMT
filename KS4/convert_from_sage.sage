import time
import numpy as np
import os.path
import uuid
import cPickle as pickle
import os
import numpy as np
top_files = [d for d in os.listdir('top_data') if d[0] == 'h' and 'topdata.p' in d]
print('num files', len(top_files))
assert_mismatch = []
global count_all
def top_info_from_file(file):
    h11 = int(file.split("_")[1])
    poly = int(file.split("_poly")[1].split("_topdata")[0])
#     s = file.split("_poly")[2]
#     poly = int(
#     s[s.index('y')+1:])
    f = open("top_data/"+file,'rb')
    data = pickle.load(f)
    f.close()
    verts0 = data[1]
    verts1  = [[int(j) for j in vert] for vert in verts0]
    genera0 = data[2]
    genera1  = [int(j) for j in genera0]
    dualdim0 = data[3]
    dualdim1  = [int(j) for j in dualdim0]
    print file
    pickle.dump([verts1,genera1,dualdim1],open("top_data/" + file.replace("topdata","topdata_ints"),'w'))
    #print verts0, genera0, dualdim
    #pickle.dump(str(data), )
    #to_ret = [(h11,poly),evals]
#     try:
#         assert len(evals) == h11
#     except AssertionError:
#         assert_mismatch.append(to_ret[0])
    #return to_ret
counter = 0
for file in top_files:
	print counter
	top_info_from_file(file)
	counter +=1
#raw_top_data = [top_info_from_file(file) for file in top_files]
# print(set([len(r[1]) - r[0][0] for r in raw_data]))
# raw_data = [r for r in raw_data if len(r[1]) - r[0][0] == 0]
# print("num files after eval mismatch subtraction", len(raw_data))
# print(assert_mismatch)