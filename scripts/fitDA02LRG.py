import BAOfit as bf
import numpy as np
import os
from matplotlib import pyplot as plt


Nmock = 500

import argparse

import pycorr

parser = argparse.ArgumentParser()
parser.add_argument("--zmin", help="minimum redshift",default=0.8,type=float)
parser.add_argument("--zmax", help="maximum redshift",default=1.1,type=float)
parser.add_argument("--dataver", help="data version",default='test')
parser.add_argument("--njack", help="number of jack knife used",default='60')
parser.add_argument("--weight", help="weight type used for xi",default='default')
parser.add_argument("--reg", help="regions used for xi",default='NScomb_')
parser.add_argument("--dperp", help="transverse damping; default is about right for z~1",default=4.0,type=float)
parser.add_argument("--drad", help="radial damping; default is about right for z~1",default=8.0,type=float)
parser.add_argument("--sfog", help="streaming velocity term; default standardish value",default=3.0,type=float)
parser.add_argument("--beta", help="fiducial beta in template; shouldn't matter for pre-rec",default=0.4,type=float)
parser.add_argument("--gentemp", help="whether or not to generate BAO templates",default=True,type=bool)
parser.add_argument("--gencov", help="whether or not to generate cov matrix",default=True,type=bool)
parser.add_argument("--rectype", help="type of reconstruction",default=None)
args = parser.parse_args()

rmin = 50
rmax = 150
maxb = 80.
binc = 0

zmin = args.zmin
zmax = args.zmax
bs = 4


if args.gentemp:
    #make BAO template given parameters above, using DESI fiducial cosmology and cosmoprimo P(k) tools
    #mun is 0 for pre rec
    #sigs is only relevant if mun != 0 and should then be the smoothing scale for reconstructions
    #beta is b/f, so should be changed depending on tracer
    #sp is the spacing in Mpc/h of the templates that get written out, most of the rest of the code assumes 1
    #BAO and nowiggle templates get written out for xi0,xi2,xi4 (2D code reconstructions xi(s,mu) from xi0,xi2,xi4)
    if args.rectype == None:
        bf.mkxifile_3dewig(sp=1.,v='n',mun=0,beta=args.beta,sfog=args.sfog,sigt=args.dperp,sigr=args.drad,sigs=15.)
        munw = '0'
        
    elif 'iso' in args.rectype:
        bf.mkxifile_3dewig(sp=1.,v='n',mun=1,beta=args.beta,sfog=args.sfog,sigt=args.dperp,sigr=args.drad,sigs=15.)
        munw= '1'
wm = str(args.beta)+str(args.sfog)+str(args.dperp)+str(args.drad)
mod = np.loadtxt('BAOtemplates/xi0DESI'+wm+'15.0'+munw+'.dat').transpose()[1]
modsm = np.loadtxt('BAOtemplates/xi0smDESI'+wm+'15.0'+munw+'.dat').transpose()[1]



def sigreg_c12(al,chill,fac=1.,md='f'):
    #report the confidence region +/-1 for chi2
    #copied from ancient code
    chim = 1000
    
    
    chil = []
    for i in range(0,len(chill)):
        chil.append((chill[i],al[i]))
        if chill[i] < chim:
            chim = chill[i]
            am = al[i]
            im = i
    #chim = min(chil)   
    a1u = 2.
    a1d = 0
    a2u = 2.
    a2d = 0
    oa = 0
    ocd = 0
    s0 = 0
    s1 = 0
    for i in range(im+1,len(chil)):
        chid = chil[i][0] - chim
        if chid > 1. and s0 == 0:
            a1u = (chil[i][1]/abs(chid-1.)+oa/abs(ocd-1.))/(1./abs(chid-1.)+1./abs(ocd-1.))
            s0 = 1
        if chid > 4. and s1 == 0:
            a2u = (chil[i][1]/abs(chid-4.)+oa/abs(ocd-4.))/(1./abs(chid-4.)+1./abs(ocd-4.))
            s1 = 1
        ocd = chid  
        oa = chil[i][1]
    oa = 0
    ocd = 0
    s0 = 0
    s1 = 0
    for i in range(1,im):
        chid = chil[im-i][0] - chim
        if chid > 1. and s0 == 0:
            a1d = (chil[im-i][1]/abs(chid-1.)+oa/abs(ocd-1.))/(1./abs(chid-1.)+1./abs(ocd-1.))
            s0 = 1
        if chid > 4. and s1 == 0:
            a2d = (chil[im-i][1]/abs(chid-4.)+oa/abs(ocd-4.))/(1./abs(chid-4.)+1./abs(ocd-4.))
            s1 = 1
        ocd = chid  
        oa = chil[im-i][1]
    if a1u < a1d:
        a1u = 2.
        a1d = 0
    if a2u < a2d:
        a2u = 2.
        a2d = 0
            
    return am,a1d,a1u,a2d,a2u,chim  


def get_xi0cov():
    
    dirm = '/global/project/projectdirs/desi/users/dvalcin/Mocks/2PCF/'
    fnm = 'xi_lognormal_lrg_sub_'
    xin0 = np.loadtxt(dirm+fnm+'1.txt')
    nbin = len(xin0)
    print(nbin)
    xiave = np.zeros((nbin))
    cov = np.zeros((nbin,nbin))

    Ntot = 0
    fac = 1.
    for i in range(1,Nmock):
        nr = str(i)
        xii = np.loadtxt(dirm+fnm+nr+'.txt').transpose()
        xic = xii[1]
        xiave += xic
        Ntot += 1.
    print( Ntot)        
    xiave = xiave/float(Ntot)
    for i in range(1,Nmock):
        nr = str(i)
        xii = np.loadtxt(dirm+fnm+nr+'.txt').transpose()
        xic = xii[1]
        for j in range(0,nbin):
            xij = xic[j]#-angfac*xiit[j]
            for k in range(0,nbin):
                xik = xic[k]#-angfac*xiit[k]
                cov[j][k] += (xij-xiave[j])*(xik-xiave[k])

    cov = cov/float(Ntot)                   
        
    return cov



datadir =  '/global/cfs/cdirs/desi/survey/catalogs/DA02/LSS/guadalupe/LSScats/'+args.dataver+'/xi/'

#data = datadir+'xi024LRGDA02_'+str(zmin)+str(zmax)+'2_default_FKPlin'+str(bs)+'.dat'
if args.rectype == None:
    data = datadir +'/smu/xipoles_LRG_'+args.reg+str(zmin)+'_'+str(zmax)+'_'+args.weight+'_lin'+str(bs)+'_njack'+args.njack+'.txt'
else:
    data = datadir +'/smu/xipoles_LRG_'+args.rectype+args.reg+str(zmin)+'_'+str(zmax)+'_'+args.weight+'_lin'+str(bs)+'_njack'+args.njack+'.txt'
    
d = np.loadtxt(data).transpose()
xid = d[2]
rl = []
nbin = 0
for i in range(0,len(d[0])):
    r = i*bs+bs/2.+binc
    rbc = .75*((r+bs/2.)**4.-(r-bs/2.)**4.)/((r+bs/2.)**3.-(r-bs/2.)**3.) #correct for pairs should have slightly larger average pair distance than the bin center
    rl.append(rbc) 
    if rbc > rmin and rbc < rmax:
        nbin += 1
rl = np.array(rl)
print(rl)
print(xid)
covm = get_xi0cov() #will become covariance matrix to be used with data vector
cfac = 5/4
covm *= cfac**2.
diag = []
for i in range(0,len(covm)):
    diag.append(np.sqrt(covm[i][i]))
diag = np.array(diag)
plt.plot(rl,rl*diag,label='lognormal mocks')
plt.plot(rl,rl*d[5],label='jack-knife')
plt.xlabel('s (Mpc/h)')
plt.ylabel(r's$\sigma$')
plt.legend()
plt.title('apply a factor '+str(round(cfac,2))+' to the mock error')
plt.show()


spa=.001
outdir = os.environ['HOME']+'/DA02baofits/'
lik = bf.doxi_isolike(xid,covm,mod,modsm,rl,bs=bs,rmin=rmin,rmax=rmax,npar=3,sp=1.,Bp=.4,rminb=50.,rmaxb=maxb,spa=spa,mina=.8,maxa=1.2,Nmock=Nmock,v='',wo='LRG'+str(zmin)+str(zmax)+wm+str(bs),diro=outdir)
print('minimum chi2 is '+str(min(lik))+' for '+str(nbin-5)+' dof')
liksm = bf.doxi_isolike(xid,covm,modsm,modsm,rl,bs=bs,rmin=rmin,rmax=rmax,npar=3,sp=1.,Bp=.4,rminb=50.,rmaxb=maxb,spa=spa,mina=.8,maxa=1.2,Nmock=Nmock,v='',wo='LRGsm'+str(zmin)+str(zmax)+wm+str(bs),diro=outdir)
#print(lik)
#print(liksm)
al = [] #list to be filled with alpha values
for i in range(0,len(lik)):
    a = .8+spa/2.+spa*i
    al.append(a)
#below assumes you have matplotlib to plot things, if not, save the above info to a file or something

sigs = sigreg_c12(al,lik)
print('result is alpha = '+str((sigs[2]+sigs[1])/2.)+'+/-'+str((sigs[2]-sigs[1])/2.))


plt.plot(al,lik-min(lik),'k-',label='BAO template')
plt.plot(al,liksm-min(lik),'k:',label='no BAO')
plt.xlabel(r'$\alpha$ (relative isotropic BAO scale)')
plt.ylabel(r'$\Delta\chi^{2}$')
plt.legend()
plt.show()

plt.errorbar(rl,rl**2.*xid,rl**2*diag,fmt='ro')
fmod = outdir+'ximodLRG'+str(zmin)+str(zmax)+wm+str(bs)+'.dat'
mod = np.loadtxt(fmod).transpose()
plt.plot(mod[0],mod[0]**2.*mod[1],'k-')
plt.xlim(20,rmax+10)
plt.ylim(-50,100)
plt.show()
                