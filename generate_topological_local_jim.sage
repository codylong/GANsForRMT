import time
import numpy as np
import os.path

if 'data' not in vars():
        data = {
                10: sage.geometry.lattice_polytope.read_all_polytopes("h11_10.txt"),
                20: sage.geometry.lattice_polytope.read_all_polytopes("h11_20.txt"),
                30: sage.geometry.lattice_polytope.read_all_polytopes("h11_30.txt"),
                40: sage.geometry.lattice_polytope.read_all_polytopes("h11_40.txt"),
                50: sage.geometry.lattice_polytope.read_all_polytopes("h11_50.txt")
        }

def dot(v1,v2):
        p1 = list(v1)
        p2 = list(v2)
        return p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2] + p1[3]*p2[3]

def getdual(points,dualpoly):
        dualpoints = []
        for dual in dualpoly.points():
                isdual = True
                for point in points:
                        if dot(dual, point) != -1:
                                isdual = False
                if isdual:
                        dualpoints.append(dual)
        sp = LatticePolytope(dualpoints)
        return len(sp.interior_points()), sp.dim()

def getdualfav(points,dualpoly):
        dualpoints = []
        for dual in dualpoly.points():
                isdual = True
                for point in points:
                        if dot(dual, point) != -1:
                                isdual = False
                if isdual:
                        dualpoints.append(dual)
        return dualpoints

def isfavorable(npoly):
    dual = npoly.polar()
    isfavorable = True
    for face in npoly.faces_lp(2):
        dualpoints = getdualfav(face.points(),dual)
        dualface = LatticePolytope(dualpoints)
        if len(face.interior_points())*len(dualface.interior_points()) !=0:
            isfavorable = False
    return isfavorable

h11, chunk = int(sys.argv[1]), int(sys.argv[2]) ### note 10 is upper right window
print 'h11 ', h11
alldatas = []
numrays = []
import uuid
import cPickle as pickle
import os
print 'chunk is', chunk
num_ex = len(data[h11])
print 'num examples', num_ex
begin = chunk*num_ex/5
end = begin + num_ex/5
print begin, end

for pp in range(begin,end):
        p = data[h11][pp]
        print data[h11].index(p), "of", len(data[h11])
        filename = "poly"+str(data[h11].index(p))
        file1 = "/home/jim/Dropbox/Documents/RandomBergman/KS4/top_data/h11_" + str(h11) + "_"+filename+"_topdata.pickle"
        if not os.path.isfile(file1):
            pd = p.polar()
            if isfavorable(pd):
                    print 'polytope points'
                    print pd.points()
                    bpoints = [list(point) for point in pd.points() if list(point) !=[0,0,0,0]]
                    #tv = triandtoric(pd.points())
                    alldata = [data[h11].index(p)]
                    #filename = uuid.uuid4().hex[:6]
                    #outdir = "/scratch/codylong/AxionReheating/geoms" + str(ntrees) + "/" + str(filename)+ ".txt"
                    start = time.time()
                    
                    #tv =newrandomtoric(polytopetoric,ntrees)
                    alldata.append([list(v) for v in bpoints])
                    #alldata.append([list(c.ambient_ray_indices()) for c in tv.fan().cones(4)])
                    alldata.append([getdual([v],p)[0] for v in bpoints])
                    alldata.append([getdual([v],p)[1] for v in bpoints])
                    end = time.time()
                    pickle.dump( alldata, open( file1, "w" ) )

                #get a basis of 1,1 forms to do intersection theory with
              
import datetime
print datetime.datetime.now()