#environment details:
#source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
#PYTHONPATH=$PYTHONPATH:$HOME/code/BAOfit_xs ; replace $HOME/code/ with wherever you cloned it

import BAOfit as bf
import numpy as np
import os
import sys
from matplotlib import pyplot as plt
import numpy.linalg as linalg

import argparse

import pycorr

parser = argparse.ArgumentParser()
parser.add_argument("--zmin", help="minimum redshift",default=0.8,type=float)
parser.add_argument("--zmax", help="maximum redshift",default=1.1,type=float)
parser.add_argument("--dperp", help="transverse damping; default is about right for z~1",default=4.0,type=float)
parser.add_argument("--drad", help="radial damping; default is about right for z~1",default=8.0,type=float)
parser.add_argument("--sfog", help="streaming velocity term; default standardish value",default=3.0,type=float)

parser.add_argument("--gentemp", help="whether or not to generate BAO templates",default=True,type=bool)
parser.add_argument("--gencov", help="whether or not to generate cov matrix",default=True,type=bool)
parser.add_argument("--pv", help="whose abacus paircounts; options are CS or JM",default='CS')
parser.add_argument("--par", help="do 25 realizations in parallel",default=True,type=bool)
parser.add_argument("--statsonly", help="if True, skip everything except for stats at end",default=False,type=bool)
args = parser.parse_args()

dofit = True

if args.statsonly:
    args.gencov = False
    args.par = False
    args.gentemp = False
    dofit = False

rmin = 50
rmax = 150
rmaxb = 80.
binc = 0

zmin = args.zmin
zmax = args.zmax
bs = 4

if bs != 4:
    print('only a binsize of 4 is supported, exiting')
    sys.exit()


nbt = int(2*(rmax-rmin)/bs)

sfog = args.sfog #fog velocity term, 3 is kind of cannonical
dperp = args.dperp # 
drad = args.drad # 

Nmock = 1000



#make BAO template given parameters above, using DESI fiducial cosmology and cosmoprimo P(k) tools
#mun is 0 for pre rec
#sigs is only relevant if mun != 0 and should then be the smoothing scale for reconstructions
#beta is b/f, so should be changed depending on tracer
#sp is the spacing in Mpc/h of the templates that get written out, most of the rest of the code assumes 1
#BAO and nowiggle templates get written out for xi0,xi2,xi4 (2D code reconstructions xi(s,mu) from xi0,xi2,xi4)


if args.gentemp:
    bf.mkxifile_3dewig(sp=1.,v='n',mun=0,beta=0.4,sfog=sfog,sigt=dperp,sigr=drad,sigs=15.)

#sys.exit()

#make covariance matrix from EZ mocks
#def get_xi0cov():
if args.gencov:
    znm = str(10*zmin)[:1]+str(10*zmax)[:1]
    dirm = '/global/cfs/cdirs/desi/cosmosim/KP45/MC/Clustering/AbacusSummit/CutSky/LRG/Xi/csaulder/EZmocks/'
    fnm = 'EZmock_results_'+znm
    result = pycorr.TwoPointCorrelationFunction.load(dirm+fnm+'_1.npy')
    rebinned = result[:(result.shape[0]//bs)*bs:bs]
    ells = (0, 2)
    s, xiell = rebinned(ells=ells, return_sep=True)
    indmin = 0
    indmax = len(s)
    indmaxb = len(s)
    sm = 0
    sx = 0
    sxb = 0
    for i in range(0,len(s)):
        if s[i] > rmin and sm == 0:
            indmin = i
            sm = 1
        if s[i] > rmax and sx == 0:
            indmax = i
            sx = 1
        if s[i] > rmaxb and sxb == 0:
            indmaxb = i
            sxb = 1
    print(indmin,indmax)        
    nbin = 2*(indmax-indmin)
    print(nbin)
    xiave = np.zeros((nbin))
    cov = np.zeros((nbin,nbin))
    nbinb = 2*(indmaxb-indmin)
    print(nbinb)
    xiaveb = np.zeros((nbinb))
    covb = np.zeros((nbinb,nbinb))

    Ntot = 0
    fac = 1.
    for i in range(1,Nmock+1):
        nr = '_'+str(i)
        result = pycorr.TwoPointCorrelationFunction.load(dirm+fnm+nr+'.npy')
        rebinned = result[:(result.shape[0]//bs)*bs:bs]
        xic0 = rebinned(ells=ells)[0][indmin:indmax]
        xic2 = rebinned(ells=ells)[1][indmin:indmax]
        xic = np.concatenate((xic0,xic2))
        xiave += xic
        xic0b = rebinned(ells=ells)[0][indmin:indmaxb]
        xic2b = rebinned(ells=ells)[1][indmin:indmaxb]
        xicb = np.concatenate((xic0b,xic2b))
        xiaveb += xicb
        Ntot += 1.
    print( Ntot)        
    xiave = xiave/float(Ntot)
    xiaveb = xiaveb/float(Ntot)
    for i in range(1,Nmock+1):
        nr = '_'+str(i)
        result = pycorr.TwoPointCorrelationFunction.load(dirm+fnm+nr+'.npy')
        rebinned = result[:(result.shape[0]//bs)*bs:bs]
        xic0 = rebinned(ells=ells)[0][indmin:indmax]
        xic2 = rebinned(ells=ells)[1][indmin:indmax]
        xic = np.concatenate((xic0,xic2))
        xic0b = rebinned(ells=ells)[0][indmin:indmaxb]
        xic2b = rebinned(ells=ells)[1][indmin:indmaxb]
        xicb = np.concatenate((xic0b,xic2b))
        for j in range(0,nbin):
            xij = xic[j]
            for k in range(0,nbin):
                xik = xic[k]
                cov[j][k] += (xij-xiave[j])*(xik-xiave[k])
        for j in range(0,nbinb):
            xij = xicb[j]
            for k in range(0,nbinb):
                xik = xicb[k]
                covb[j][k] += (xij-xiaveb[j])*(xik-xiaveb[k])

    cov = cov/float(Ntot)             
    covb = covb/float(Ntot)      
    sc = np.concatenate((s[indmin:indmax],s[indmin:indmax]))
    scb = np.concatenate((s[indmin:indmaxb],s[indmin:indmaxb]))
    #return cov

    #cov = get_xi0cov()
    xistd = []
    covn = np.zeros((len(xiave),len(xiave)))
    for i in range(0,len(xiave)):
         xistd.append(np.sqrt(cov[i][i]))
         for j in range(0,len(xiave)):
             covn[i][j] = cov[i][j]/np.sqrt(cov[i][i]*cov[j][j])
    # plt.errorbar(sc,sc**2.*xiave,sc**2.*np.array(xistd))
    # plt.show()
    # invcov = linalg.inv(cov)
    # #plt.imshow(invcov)
    # plt.imshow(covn)
    # plt.show()
    # sys.exit()

    invc = np.linalg.inv(cov) #the inverse covariance matrix to pass to the code
    invcb = np.linalg.inv(covb) #the inverse covariance matrix to get the bias values to pass to the code

    print(sc)
    #This assumes sc has bin center values and then applies correction assuming spherically symmetric distribution of data
    rl = []
    #nbin = 0
    for i in range(0,len(sc)):
        r = sc[i]
        #correct for pairs should have slightly larger average pair distance than the bin center
        #this assumes mid point of bin is being used and pairs come from full 3D volume
        rbc = .75*((r+bs/2.)**4.-(r-bs/2.)**4.)/((r+bs/2.)**3.-(r-bs/2.)**3.) 
        rl.append(rbc) 

    rlb = []
    #nbin = 0
    for i in range(0,len(scb)):
        r = scb[i]
        #correct for pairs should have slightly larger average pair distance than the bin center
        #this assumes mid point of bin is being used and pairs come from full 3D volume
        rbc = .75*((r+bs/2.)**4.-(r-bs/2.)**4.)/((r+bs/2.)**3.-(r-bs/2.)**3.) 
        rlb.append(rbc) 


wm = str(sfog)+str(dperp)+str(drad)
mod = 'DESI0.4'+wm+'15.00.dat'


#bias priors, log around best fit up to rmaxb
Bp = 100#0.4
Bt = 100#0.4

spa = .001
mina = .8
maxa = 1.2
outdir = os.environ['HOME']+'/DESImockbaofits/'



if args.pv == 'JM':
    abdir = '/global/cfs/cdirs/desi/cosmosim/KP45/MC/Clustering/AbacusSummit/CutSky/LRG/Xi/jmena/'
if args.pv == 'CS':
    abdir = '/global/cfs/cdirs/desi/cosmosim/KP45/MC/Clustering/AbacusSummit/CutSky/LRG/Xi/csaulder/CF_multipoles/'



def doreal(mn):
    if args.pv == 'CS':
        fnm = abdir+'results_realization'+str(mn).zfill(3)+'_rand20_'+znm+'.dat'
        xis = np.loadtxt(fnm).transpose()
        xid0 = xis[2][indmin:indmax]
        xid2 = xis[3][indmin:indmax]
        xid0b = xis[2][indmin:indmaxb]
        xid2b = xis[3][indmin:indmaxb]
#       fnm = 'results_realization'+str(mn).zfill(3)+'_rand20_'+znm+'.npy'
#       result = pycorr.TwoPointCorrelationFunction.load(abdir+fnm)
#       rebinned = result[:(result.shape[0]//bs)*bs:bs]
#       ells = (0, 2)
#       s, xiell = rebinned(ells=ells, return_sep=True)
# 
#       xid0 = xiell[0][indmin:indmax]
#       xid2 = xiell[1][indmin:indmax]
#       
#       xid0b = xiell[0][indmin:indmaxb]
#       xid2b = xiell[1][indmin:indmaxb]

    if args.pv == 'JM':
        xid0 = np.loadtxt(abdir+'Xi_0_zmin'+str(zmin)+'_zmax'+str(zmax)+'.txt').transpose()[mn][indmin:indmax] 
        xid2 = np.loadtxt(abdir+'Xi_2_zmin'+str(zmin)+'_zmax'+str(zmax)+'.txt').transpose()[mn][indmin:indmax] 

        xid0b = np.loadtxt(abdir+'Xi_0_zmin'+str(zmin)+'_zmax'+str(zmax)+'.txt').transpose()[mn][indmin:indmaxb] 
        xid2b = np.loadtxt(abdir+'Xi_2_zmin'+str(zmin)+'_zmax'+str(zmax)+'.txt').transpose()[mn][indmin:indmaxb] 

    xid = np.concatenate((xid0,xid2))
    xidb = np.concatenate((xid0b,xid2b))    
    fout = 'LRGabcutsky0_'+args.pv+str(zmin)+str(zmax)+wm+'_real'+str(mn)+'_'+str(bs)
    bf.Xism_arat_1C_an(xid,invc,rl,mod,xidb,invcb,rlb,verbose=True,Bp=Bp,Bt=Bt,fout=fout,dirout=outdir)
    #bf.plot_2dlik(os.environ['HOME']+'/DESImockbaofits/2Dbaofits/arat'+fout+'1covchi.dat')
    #modl = np.loadtxt(outdir+'ximod'+fout+'.dat').transpose()
    #plt.errorbar(sc,sc**2.*xid,sc**2.*xistd,fmt='ro')
    #plt.plot(sc,sc**2.*modl[1],'k-')
    #plt.show()

if dofit:
    if args.par:
        from multiprocessing import Pool
        N = 25
        p = Pool(N)
        inds = np.arange(N)
        p.map(doreal,inds)

    else:
        doreal(0)

#compile stats
Nmock = 25
foutall = outdir+'AperpAparfits_LRGabcutsky0_'+args.pv+str(zmin)+str(zmax)+wm+'_'+str(bs)+'.txt'
fo = open(foutall,'w')
fo.write('#Mock_number <alpha_||> sigma(||) <alpha_perp> sigma_perp min(chi2) cov_||,perp corr_||,perp\n')
for ii in range(0,Nmock):
    fout = 'LRGabcutsky0_'+args.pv+str(zmin)+str(zmax)+wm+'_real'+str(ii)+'_'+str(bs)
    ans = bf.sigreg_2dEZ(outdir+'2Dbaofits/arat'+fout+'1covchi.dat')
    fo.write(str(ii)+' ')
    for val in ans:
        fo.write(str(val)+' ')
    fo.write('\n')
fo.close()

all = np.loadtxt(foutall).transpose()

meanchi2 = np.mean(all[-3])
print('<chi2>/dof is '+str(round(meanchi2,3))+'/'+str(nbt-10))

meanapar = np.mean(all[1])
stdapar = np.std(all[1])
meanspar = np.mean(all[2])
print('<alpha_||>,std(alpha_||),<sigma_||>')
print(meanapar,stdapar,meanspar)
meanaperp = np.mean(all[3])
stdaperp = np.std(all[3])
meansperp = np.mean(all[4])
print('<alpha_perp>,std(alpha_perp),<sigma_perp>')
print(meanaperp,stdaperp,meansperp)
corrparperp = (np.sum(all[1]*all[3])/Nmock-meanaperp*meanapar)/(stdaperp*stdapar)
meancorr = np.mean(all[-1])
print('corr_par,perp,<corr_par,perp>')
print(corrparperp,meancorr)

    


sys.exit()

