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

def triandtoric(pts):
    PointConfiguration.set_engine('topcom')
    LP = LatticePolytope(pts)
    facets = LP.facets_lp()
    ptlist = list([list(l) for l in LP.points()])
    PC = PointConfiguration(ptlist)
    ptspre = [list(point) for point in PC.points()]
    PC2 = PointConfiguration(ptspre).exclude_points(PC.face_interior(codim = 1)).restrict_to_regular_triangulations().restrict_to_fine_triangulations()
    pts = [list(point) for point in PC2.points()]
    facetriang = []
    triang  = eval(str(PC2.triangulate()).replace('<','[').replace('>',']'))
    t2 = [  [list(pts[simp[0]]),list(pts[simp[1]]),list(pts[simp[2]]),list(pts[simp[3]]), list(pts[simp[4]])] for simp in triang]
    for simplex in t2:
        for facet in facets:
                facetpts = [list(p) for p in facet.points()]
                overlap = [x for x in simplex if x in facetpts]
                if len(overlap) == 4:
                    facetriang.append(overlap)
    cones = [Cone(c) for c in facetriang]
    fan = Fan(cones)
    V = ToricVariety(fan)
    return V

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

for pp in range(737,end):
        p = data[h11][pp]
        print data[h11].index(p), "of", len(data[h11])
        filename = "poly"+str(data[h11].index(p))
        file1 = "/home/jim/Dropbox/Documents/RandomBergman/KS4/top_data/h11_" + str(h11) + "_"+filename+"_topdata_more.pickle"
        #file1 = "/Users/cody/Dropbox/RandomBergman/KS4/top_data/h11_" + str(h11) + "_"+filename+"_topdata_more.pickle"
        #file1 = "/scratch/codylong/KS4/data//h11_" + str(h11) + "_"+filename+"_topdata_more.pickle"
        if not os.path.isfile(file1):
            pd = p.polar()
            if isfavorable(pd):
                    print 'polytope points'
                    print pd.points()
                    bpoints = [list(point) for point in pd.points() if list(point) !=[0,0,0,0]]
                    #tv = triandtoric(pd.points())
                    
                    #filename = uuid.uuid4().hex[:6]
                    #outdir = "/scratch/codylong/AxionReheating/geoms" + str(ntrees) + "/" + str(filename)+ ".txt"
                    
                    #tv =newrandomtoric(polytopetoric,ntrees)
                    #alldata.append([list(v) for v in bpoints])
                    #alldata.append([list(c.ambient_ray_indices()) for c in tv.fan().cones(4)])

                    tv = triandtoric(pd.points())
                    

                    #divisor topologies
                    pairs = []
                    for v in bpoints:
                        pairs.append(getdual([v],p))
                    

                    
                    #tv =newrandomtoric(polytopetoric,ntrees)
                    #alldata.append([list(v) for v in tv.fan().rays()])
                    #alldata.append([list(c.ambient_ray_indices()) for c in tv.fan().cones(4)])

                    #get a basis of 1,1 forms to do intersection theory with
                    b = tv.cohomology_basis(1)
                    gens = tv.cohomology_ring().gens()

                    zstring=str(b).replace('[','').replace(']','')
                    #print zstring.replace('(','{').replace(')','}')
                    HH = tv.cohomology_ring()
                    hyper = -HH(tv.K())
                    intersecting = b
                    
                    J = var(','.join('J%s'%i for i in range(1,len(intersecting)+1)))
                    D = var(','.join('D%s'%i for i in range(1,len(intersecting)+1)))

                    #alldata.append(J)

                    #print "Corresponding basis of divisors is:"
                    #print str(D).replace('(','{').replace(')','}')

                    # f = open(outdir, 'a')
                    # f.write("Corresponding basis of divisors is:\n" + str(D).replace('(','{').replace(')','}') + "\n")
                    # f.close()
                    #alldata.append(D)

                    #this map is convenient for changing basis
                    basis_map = []
                    for i,ii in enumerate(intersecting):
                            basis_map.append("["+str(ii) + "," + str(J[i]) + "]")
                    basis_map =eval(str(basis_map).replace('[z','').replace('],',',').replace('\'',''))
                    bmz = [basis_map[i][0] for i in range(len(basis_map))]
                    bmj = [basis_map[i][1] for i in range(len(basis_map))]

                    #get the linear equivilences so we can determine the extra gauge group axions
                    newgens = str(gens)
                    for m in basis_map:
                            whichd = str(m[1]).split('J')[1]
                            newgens = newgens.replace('z'+str(m[0]),'D' +whichd )
                    newgens = newgens.replace('[','').replace(']','').replace('(','{').replace(')','}')
                    #print "E8 gauge group divisors are"
                    #print newgens

                    # f = open(outdir, 'a')
                    # f.write("All toric divisors are :\n" + str(newgens).replace('(','{').replace(')','}') + "\n")
                    # f.close()
                    #alldata.append(newgens)

                    #old volume, can we do it better?

                    start = time.time()
                    volume = 0
                    for i,ii in enumerate(intersecting):
                       for j,jj in enumerate(intersecting):
                               for k,kk in enumerate(intersecting):
                                       volume = volume + tv.integrate(hyper*ii*jj*kk)*J[i]*J[j]*J[k]
                    volume = volume/6

        

                    

                    #again, can we do better?
                    curves = []
                    twocones = tv.fan().cones(3)
                    for cone in twocones:
                            inds = cone.ambient_ray_indices()
                            curvevol = 0
                            for i in range(len(intersecting)):
                                    curvevol = curvevol + tv.integrate(gens[inds[0]]*gens[inds[1]]*gens[inds[2]]*intersecting[i])*J[i]
                            curves.append(curvevol)
                    curves = list(set(curves))

                     
                    mori = curves 

                    V = volume
                    moricoeff = [vector(mori).diff(J[k]) for k in range(len(J))]
                    moricoeff = [list(mc) for mc in moricoeff]
                    moricoeff = np.array(moricoeff).T.tolist()
                    bigcurve = [0 for ii in range(len(J))]
                    for ii in range(len(J)):
                            for jj in range(len(moricoeff)):
                                    bigcurve[ii] += -moricoeff[jj][ii]
                    #find that point
                    try:
                            plin = MixedIntegerLinearProgram()
                            x = plin.new_variable(real=True, nonnegative=False)
                            obj = 0
                            for ii in range(len(bigcurve)):
                                    obj = obj + bigcurve[ii]*x[ii]
                            plin.set_objective(obj)
                            for ii in range(len(moricoeff)):
                                    curve = 0
                                    for jj in range(len(J)):
                                            curve = curve + moricoeff[ii][jj]*x[jj]
                                    plin.add_constraint(curve >= 1)
                            plin.solve()

                            #get the values of J that minimize sum_i c_i 
                            vals = plin.get_values(x)
                            Jvals = []
                            for ii in range(len(vals)):
                                    Jvals.append(vals[ii])
                            Jrep = {}
                            #create a dictionary to easily plug in for J at these values
                            for ii in range(len(J)):
                                    Jrep[J[ii]] = Jvals[ii]
                            #get overall and curve volumes
                            V0 = V.substitute(Jrep)

                            #normalize to volume = 1:
                            Jrep = {}
                            #create a dictionary to easily plug in for J at these values
                            for ii in range(len(J)):
                                    Jrep[J[ii]] = Jvals[ii]/V0**(1/3)
                            #get overall and curve volumes
                            V0 = V.substitute(Jrep)


                            cvals = [mori[ii].substitute(Jrep) for ii in range(len(mori))]


                            #get divisor volumes and metric
                            tau = [(V.diff(J[ii])) for ii in range(len(J))]
                            tau0 = [tau[ii].substitute(Jrep) for ii in range(len(J))]
                            tau0 = [np.float(ta) for ta in tau0]
                            Kinv0 = [[(-4.0*V0)*((tau[ii].diff(J[jj])).substitute(Jrep)) for jj in range(len(J))] for ii in range(len(J))]
                            Ainv0 = np.array([[np.float64(Kinv0[ii][jj]) for ii in range(len(J))] for jj in range(len(J))])
                            for ii in range(len(J)):
                                    for jj in range(len(J)):
                                            Kinv0[ii][jj] += 4.0*tau0[ii]*tau0[jj]
                            Kinv0lst = np.array([[np.float64(Kinv0[ii][jj]) for ii in range(len(J))] for jj in range(len(J))])
                            
                            #get eigenvalues and eigenvectors
                            evals, evecs = np.linalg.eig(Kinv0lst)

                            Aevals, Aevecs = np.linalg.eig(Ainv0)



                            # endall = time.time()
                            # timesall.append(endall-startall)

                            # f = open("/scratch/jhhalverson/KS4/timescaling/h11_" + str(h11) + "_"+filename+".txt",'w')
                            # f.write(str(timestoric[-1])+","+str(timescohomology[-1])+","+str(timesvolume[-1])+","+str(timescurves[-1])+","+str(timesmip[-1])+","+str(timesall[-1]))
                            # f.close()

                            # f2 = open("/scratch/jhhalverson/KS4/data/h11_" + str(h11) + "_"+filename+"_K.pickle",'w')
                            # pickle.dump(Kinv0lst,f2)
                            # f2.close()

                            # f2 = open("/scratch/jhhalverson/KS4/data/h11_" + str(h11) + "_"+filename+"_evals.pickle",'w')
                            # pickle.dump(evals,f2)
                            # f2.close()

                    except: # MIPSolverException:
                            print 'no mips solution'
                    alldata = [data[h11].index(p)]
                    #alldata.append([list(v) for v in tv.fan().rays()])
                    alldata.append(pairs)
                    #alldata.append(V0)
                    alldata.append(tau0)
                    alldata.append([float(a) for a in Aevals])
                    alldata.append([float(e) for e in evals])
                    alldata.append(len(tv.fan().cones(2)))
                    alldata.append(len(tv.fan().cones(3)))
                    alldata.append(len(tv.fan().cones(4)))
                    #alldata.append(moricoeff)  

                    print alldata

                    pickle.dump( alldata, open( file1, "w" ) )

                #get a basis of 1,1 forms to do intersection theory with
              
import datetime
print datetime.datetime.now()
