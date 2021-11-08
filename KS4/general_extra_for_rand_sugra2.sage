import time
import numpy as np

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

if 'data' not in vars():
        data = {
                10: sage.geometry.lattice_polytope.read_all_polytopes("/home/codylong/KS4/h11_10.txt"),
                15: sage.geometry.lattice_polytope.read_all_polytopes("/home/codylong/KS4/h11_15.txt"),
                20: sage.geometry.lattice_polytope.read_all_polytopes("/home/codylong/KS4/h11_20.txt"),
                30: sage.geometry.lattice_polytope.read_all_polytopes("/home/codylong/KS4/h11_30.txt"),
                40: sage.geometry.lattice_polytope.read_all_polytopes("/home/codylong/KS4/h11_40.txt"),
                50: sage.geometry.lattice_polytope.read_all_polytopes("/home/codylong/KS4/h11_50.txt")
                # 10: sage.geometry.lattice_polytope.read_all_polytopes("h11_10.txt"),
                # 20: sage.geometry.lattice_polytope.read_all_polytopes("h11_20.txt"),
                # 30: sage.geometry.lattice_polytope.read_all_polytopes("h11_30.txt"),
                # 40: sage.geometry.lattice_polytope.read_all_polytopes("h11_40.txt"),
                # 50: sage.geometry.lattice_polytope.read_all_polytopes("h11_50.txt")
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
        return dualpoints

def isfavorable(npoly):
    dual = npoly.polar()
    isfavorable = True
    for face in npoly.faces_lp(2):
        dualpoints = getdual(face.points(),dual)
        dualface = LatticePolytope(dualpoints)
        if len(face.interior_points())*len(dualface.interior_points()) !=0:
            isfavorable = False
    return isfavorable

h11, starter, num_ex = int(sys.argv[1]), int(sys.argv[2]) ,int(sys.argv[3]) ### note 10 is upper right window
print 'h11 starter num_ex', h11, starter, num_ex
p = data[h11][0]
alldatas = []
timestoric, timescohomology, timesvolume, timesall, timescurves, timesmip = [], [], [], [], [], []
numrays = []
import uuid
import cPickle as pickle
import os
assert len(data[h11][starter:starter+num_ex]) == num_ex
for p in data[h11][starter:starter+num_ex]:
        #print data[h11].index(p), "of", len(data[20])
        pd = p.polar()
        filename = "poly"+str(data[h11].index(p))
        file1 = "/scratch/codylong/KS4/data/h11_" + str(h11) + "_"+filename+"_evals.pickle"

        exists1= os.path.isfile(file1)
        if exists1 == False and isfavorable(pd):
        #if isfavorable(pd):
                print 'polytope points'
                print p.points()
                startall = time.time()
                start = time.time()
                tv = triandtoric(pd.points())
                alldata = [starter+data[h11][starter:starter+num_ex].index(p)]
                #filename += uuid.uuid4().hex[:6]
                #outdir = "/scratch/codylong/AxionReheating/geoms" + str(ntrees) + "/" + str(filename)+ ".txt"
                start = time.time()
                
                #tv =newrandomtoric(polytopetoric,ntrees)
                alldata.append([list(v) for v in tv.fan().rays()])
                alldata.append([list(c.ambient_ray_indices()) for c in tv.fan().cones(4)])
                end = time.time()
                timestoric.append(end-start)

                #get a basis of 1,1 forms to do intersection theory with
                start = time.time()
                b = tv.cohomology_basis(1)
                gens = tv.cohomology_ring().gens()
                end = time.time()
                timescohomology.append(end-start)
                numrays.append(len(tv.fan().rays()))

                zstring=str(b).replace('[','').replace(']','')
                #print zstring.replace('(','{').replace(')','}')
                HH = tv.cohomology_ring()
                hyper = -HH(tv.K())
                intersecting = b
                #for ii in b:
                #       if str(hyper*ii) != "[0]":
                #               intersecting.append(ii)
                J = var(','.join('J%s'%i for i in range(1,len(intersecting)+1)))
                D = var(','.join('D%s'%i for i in range(1,len(intersecting)+1)))

                alldata.append(J)

                #print "Corresponding basis of divisors is:"
                #print str(D).replace('(','{').replace(')','}')

                # f = open(outdir, 'a')
                # f.write("Corresponding basis of divisors is:\n" + str(D).replace('(','{').replace(')','}') + "\n")
                # f.close()
                alldata.append(D)

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
                alldata.append(newgens)

                #old volume, can we do it better?

                start = time.time()
                volume = 0
                for i,ii in enumerate(intersecting):
                   for j,jj in enumerate(intersecting):
                           for k,kk in enumerate(intersecting):
                                   volume = volume + tv.integrate(hyper*ii*jj*kk)*J[i]*J[j]*J[k]
                volume = volume/6

                end = time.time()

                timesvolume.append(end-start)

                alldata.append(volume)

                #again, can we do better?
                start = time.time()
                curves = []
                twocones = tv.fan().cones(3)
                for cone in twocones:
                        inds = cone.ambient_ray_indices()
                        curvevol = 0
                        for i in range(len(intersecting)):
                                curvevol = curvevol + tv.integrate(gens[inds[0]]*gens[inds[1]]*gens[inds[2]]*intersecting[i])*J[i]
                        curves.append(curvevol)
                curves = list(set(curves))
                end = time.time()
                timescurves.append(end-start)
                alldata.append(curves)   
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
                #try:
                start = time.time()
                p = MixedIntegerLinearProgram()
                x = p.new_variable(real=True, nonnegative=False)
                obj = 0
                for ii in range(len(bigcurve)):
                        obj = obj + bigcurve[ii]*x[ii]
                p.set_objective(obj)
                for ii in range(len(moricoeff)):
                        curve = 0
                        for jj in range(len(J)):
                                curve = curve + moricoeff[ii][jj]*x[jj]
                        p.add_constraint(curve >= 1)
                p.solve()
                end = time.time()
                timesmip.append(end-start)
                #get the values of J that minimize sum_i c_i 
                vals = p.get_values(x)
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
                Kinv0 = [[(-4.0*V0)*((tau[ii].diff(J[jj])).substitute(Jrep)) for jj in range(len(J))] for ii in range(len(J))]
                for ii in range(len(J)):
                        for jj in range(len(J)):
                                Kinv0[ii][jj] += 4.0*tau0[ii]*tau0[jj]
                Kinv0lst = np.array([[np.float64(Kinv0[ii][jj]) for ii in range(len(J))] for jj in range(len(J))])
                
                # #get eigenvalues and eigenvectors
                # evals, evecs = np.linalg.eig(Kinv0lst)

                # print 'num evals - h11', len(evals)-h11

                # strw = str(list(evals))
                # alldatas.append(alldata)

                # #now into the hessian data for sugra. there are a few pieces. first we need the overall volume, which we computed above
                # #then we need the kahler connection:
                # ka = [-np.float64(jp.substitute(Jrep)/(2*V0)) for jp in J]
                # print ka
                # J0 = [np.float64(jp.substitute(Jrep)) for jp in J]

                # #we already have the metric, so now we need the three index thing. first, construbtion the Amatrix with downstaris
                # #indices, that actually appears in the metric:
                # Aab = np.linalg.inv(np.array([[np.float64(((tau[ii].diff(J[jj])).substitute(Jrep))) for jj in range(len(J))] for ii in range(len(J))]))
                # Aabinv = np.linalg.inv(Aab)
                # delta = np.identity(len(J))
                # kappa = [[[np.float64(((V.diff(J[ii])).diff(J[jj])).diff(J[kk])) for ii in range(len(J))] for jj in range(len(J))] for kk in range(len(J))]
                # #print kappa

                # Sabc = [[[
                #  np.tensordot(np.tensordot(np.tensordot(kappa,Aab[a],[2,0]),Aab[b],[1,0]),Aab[c],[0,0])
                # for a in range(h11)] for b in range(h11)] for c in range(h11)]
                # #print Sabc[1][2][3], Sabc[3][2][1], Sabc[2][3][1]
                # kabc = [[[1/8*((Aab[a][b]*J0[c])/2 + (Aab[a][c]*J0[b])/2 + (Aab[b][c]*J0[a])/2 -J0[a]*J0[b]*J0[c]/2 +
                #  Sabc[a][b][c])
                # for a in range(h11)] for b in range(h11)] for c in range(h11)]

                # kabcd = 3/64*np.einsum('i,j,k,l',J0,J0,J0,J0)
                # -3/16*1/24*(4*np.einsum('ij,k,l',Aab,J0,J0) + 4*np.einsum('ik,j,l',Aab,J0,J0) + 4*np.einsum('il,j,k',Aab,J0,J0)
                #     + 4*np.einsum('jk,i,l',Aab,J0,J0)+ 4*np.einsum('jl,i,k',Aab,J0,J0) + 4*np.einsum('kl,i,j',Aab,J0,J0) )
                # +3/32*1/24*(4*np.einsum('ij,kl',Aab,Aab) + 4*np.einsum('ik,jl',Aab,Aab) + 4*np.einsum('il,jk',Aab,Aab)
                #     + 4*np.einsum('jk,il',Aab,Aab)+ 4*np.einsum('jl,ik',Aab,Aab) + 4*np.einsum('kl,ij',Aab,Aab) )
                # -1/8*(np.einsum('i,jkl',J0,Sabc)+ np.einsum('j,ikl',J0,Sabc)+ np.einsum('k,jil',J0,Sabc)+ np.einsum('l,jki',J0,Sabc))
                # -3/16*1/24*(4*np.einsum('mij,kln,mn',Sabc,Sabc,Aabinv) + 4*np.einsum('mik,jln,mn',Sabc,Sabc,Aabinv)
                #     +4*np.einsum('mil,kjn,mn',Sabc,Sabc,Aabinv)+4*np.einsum('mkj,iln,mn',Sabc,Sabc,Aabinv)
                #     +4*np.einsum('mlj,kin,mn',Sabc,Sabc,Aabinv)+4*np.einsum('mlk,ijn,mn',Sabc,Sabc,Aabinv))



                
                # #now connstruct the riemann curvature tensor on moduli space. let's do it in pieces:
                # Rabcd1 = [[[[1/64*J0[a]*J0[b]*J0[c]*J0[d]
                # for a in range(h11)] for b in range(h11)] for c in range(h11)] for d in range(h11)]

                # Rabcd2 = [[[[-1/64*(( Aab[a][b]*J0[c]*J0[d]+ Aab[a][d]*J0[c]*J0[b]
                #     + Aab[b][c]*J0[a]*J0[d]  + Aab[c][d]*J0[a]*J0[b]))
                # for a in range(h11)] for b in range(h11)] for c in range(h11)] for d in range(h11)]

                # Rabcd3 = [[[[3/32*(( Aab[a][b]*Aab[c][d]+Aab[a][d]*Aab[b][c]))
                # for a in range(h11)] for b in range(h11)] for c in range(h11)] for d in range(h11)]

                # SAinvS = np.dot(np.dot(Sabc,Aabinv),Sabc)

                # Rabcd4 = [[[[-1/16*(( SAinvS[a][b][d][c] + SAinvS[a][d][b][c]))
                # for a in range(h11)] for b in range(h11)] for c in range(h11)] for d in range(h11)]

                # Rabcd = [[[[Rabcd1[a][b][c][d] + Rabcd2[a][b][c][d] + Rabcd3[a][b][c][d] + Rabcd4[a][b][c][d]
                # for a in range(h11)] for b in range(h11)] for c in range(h11)] for d in range(h11)]

                # endall = time.time()
                # timesall.append(endall-startall)
                # print timesall
                #f = open("/scratch/codylong/KS4/timescaling/h11_" + str(h11) + "_"+filename+".txt",'w')
                #f.write(str(timestoric[-1])+","+str(timescohomology[-1])+","+str(timesvolume[-1])+","+str(timescurves[-1])+","+str(timesmip[-1])+","+str(timesall[-1]))
                #f.close()

                f2 = open("/scratch/codylong/KS4/data/h11_" + str(h11) + "_"+filename+"_K.pickle",'w')
                pickle.dump(Kinv0lst,f2)
                f2.close()

                # f2 = open("/scratch/codylong/KS4/data/h11_" + str(h11) + "_"+filename+"_evals.pickle",'w')
                # pickle.dump(evals,f2)
                # f2.close()

                # f2 = open("/scratch/codylong/KS4/data/h11_" + str(h11) + "_"+filename+"_ka.pickle",'w')
                # pickle.dump(ka,f2)
                # f2.close()

                # f2 = open("/scratch/codylong/KS4/data/h11_" + str(h11) + "_"+filename+"_kabc.pickle",'w')
                # pickle.dump(kabc,f2)
                # f2.close()

                # f2 = open("/scratch/codylong/KS4/data/h11_" + str(h11) + "_"+filename+"_kabcd.pickle",'w')
                # pickle.dump(kabcd,f2)
                # f2.close()

                        # f = open("/Users/cody/Dropbox/tempdump//h11_" + str(h11) + "_"+filename+".txt",'w')
                        # f.write(str(timestoric[-1])+","+str(timescohomology[-1])+","+str(timesvolume[-1])+","+str(timescurves[-1])+","+str(timesmip[-1])+","+str(timesall[-1]))
                        # f.close()

                        # f2 = open("/Users/cody/Dropbox/tempdump//h11_" + str(h11) + "_"+filename+"_K.pickle",'w')
                        # pickle.dump(Kinv0lst,f2)
                        # f2.close()

                        # f2 = open("/Users/cody/Dropbox/tempdump//h11_" + str(h11) + "_"+filename+"_evals.pickle",'w')
                        # pickle.dump(evals,f2)
                        # f2.close()

                        # f2 = open("/Users/cody/Dropbox/tempdump//h11_" + str(h11) + "_"+filename+"_ka.pickle",'w')
                        # pickle.dump(ka,f2)
                        # f2.close()

                        # f2 = open("/Users/cody/Dropbox/tempdump//h11_" + str(h11) + "_"+filename+"_kabc.pickle",'w')
                        # pickle.dump(kabc,f2)
                        # f2.close()

                        # f2 = open("/Users/cody/Dropbox/tempdump//h11_" + str(h11) + "_"+filename+"_kabcd.pickle",'w')
                        # pickle.dump(kabcd,f2)
                        # f2.close()



                #except: # MIPSolverException:
                 #       print 'no mips solution'
            
# import datetime
# print datetime.datetime.now()
