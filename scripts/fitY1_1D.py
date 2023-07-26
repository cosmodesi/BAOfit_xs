import BAOfit as bf
import numpy as np
import os
import sys
from matplotlib import pyplot as plt


Nmock = 1000

import argparse

import pycorr

parser = argparse.ArgumentParser()
parser.add_argument("--tracer", help="tracer type",default='LRG')
parser.add_argument("--zmin", help="minimum redshift",default=0.4,type=float)
parser.add_argument("--zmax", help="maximum redshift",default=0.6,type=float)
parser.add_argument("--rmin", help="minimum redshift",default=50,type=float)
parser.add_argument("--rmax", help="maximum redshift",default=150,type=float)

#parser.add_argument("--bs", help="bin size in Mpc/h, some integer multiple of 1",default=4,type=int)
parser.add_argument("--cfac", help="any factor to apply to the cov matrix",default=1,type=float)
parser.add_argument("--diagfac", help="apply a factor to only the diagonal of the cov matrix",default=1,type=float)
parser.add_argument("--catver", help="data version",default='v0.1')
parser.add_argument("--njack", help="number of jack knife used",default='0')
parser.add_argument("--weight", help="weight type used for xi",default='default_FKP')
parser.add_argument("--reg", help="regions used for xi",default='GCcomb_')
parser.add_argument("--dperp", help="transverse damping; default is about right for z~1",default=4.0,type=float)
parser.add_argument("--drad", help="radial damping; default is about right for z~1",default=8.0,type=float)
parser.add_argument("--sfog", help="streaming velocity term; default standardish value",default=3.0,type=float)
parser.add_argument("--beta", help="fiducial beta in template; shouldn't matter for pre-rec",default=0.4,type=float)
parser.add_argument("--gentemp", help="whether or not to generate BAO templates",default=True,type=bool)
parser.add_argument("--covmd", help="what type of cov matrix to use",default='RascalC')
parser.add_argument("--rectype", help="type of reconstruction",default=None)
parser.add_argument("--smooth", help="smoothing in reconstruction reconstruction",choices=['','/recon_sm10','/recon_sm15'],default='')
parser.add_argument("--covrec", help="type of reconstruction used for cov generation",default='')
parser.add_argument("--covver", help="version associated with the covariance matrix",default='v0.1')
parser.add_argument("--blinded", help="whether to use blinded catalogs",default='/blinded')
args = parser.parse_args()

rmin = args.rmin
rmax = args.rmax
maxb = 80.
binc = 0

zmin = args.zmin
zmax = args.zmax
zr = str(zmin)+'_'+str(zmax)
bs = 4#args.bs
cov_rmin = 20

#if args.rectype is None:
#    dirxi = '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/'+args.catver+blinded+'/xi/smu/'
#else:
dirxi = '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/'+args.catver+args.blinded+args.smooth+'/xi/smu/'

dircov = '/global/cfs/cdirs/desi/users/mrash/RascalC/Y1'+args.blinded+'/'+args.covver

outdir = '/global/cfs/cdirs/desi/science/Y1KP/BAO/'+args.catver+args.blinded+args.smooth+'/AJR/'

os.makedirs(outdir,exist_ok = True)

if args.covmd == 'RascalC':
    covf = dircov+'/xi024_'+args.tracer+args.covrec+'_'+args.reg+'_'+zr+'_default_FKP_lin4_s20-200_cov_RascalC_Gaussian.txt'
else:
    sys.exit('RascalC is the only coded option so far')

df = dirxi+'xipoles_'+tp+args.rectype+'_'+reg+'_'+zr+'_default_FKP_lin4_njack0_nran4_split20.txt'
print('using '+df+' for data vector')
dxi = np.loadtxt(df).transpose()
print('using '+covf+' for cov matrix')
cov = np.loadtxt(covf)
#except:
#    print('not using rec cov')
#    cov = np.loadtxt(dircov+'xi024_'+tp+'_'+reg+'_'+zr+'_default_FKP_lin4_s20-200_cov_RascalC_Gaussian.txt')


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
    elif 'sym' in args.rectype:
        bf.mkxifile_3dewig(sp=1.,v='n',mun=0,beta=args.beta,sfog=args.sfog,sigt=args.dperp,sigr=args.drad,sigs=15.)
        munw = '0'
            
    elif 'iso' in args.rectype:
        bf.mkxifile_3dewig(sp=1.,v='n',mun=1,beta=args.beta,sfog=args.sfog,sigt=args.dperp,sigr=args.drad,sigs=15.)
        munw= '1'
    else:
        sys.exit(str(args.rectype)+ ' was not a currently valid choice')
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

ells = 0

if cov_rmin == 20:	
    cov = cov[:45,:45]
    diag = np.zeros(len(cov))
    rl = dxi[0][5:]
    xid = dxi[2][5:]

for i in range(0,len(cov)):
	diag[i] = np.sqrt(cov[i][i])
bs = 4

flout = args.tracer+zr+wm+str(bs)+'_cov'+args.covmd+args.covver+args.covrec

lik = bf.doxi_isolike(xid,cov,mod,modsm,rl,bs=bs,rmin=rmin,rmax=rmax,npar=3,sp=1.,Bp=.4,rminb=rmin,rmaxb=maxb,spa=spa,mina=.8,maxa=1.2,Nmock=Nmock,v='',wo=flout,diro=outdir)
selr = rl > rmin
selr &= rl < rmax
nbin = len(rl[selr])
minchi2 = min(lik)
print('minimum chi2 is '+str(min(lik))+' for '+str(nbin-5)+' dof')
print('doing no BAO fit')
liksm = bf.doxi_isolike(xid,cov,modsm,modsm,rl,bs=bs,rmin=rmin,rmax=rmax,npar=3,sp=1.,Bp=.4,rminb=rmin,rmaxb=maxb,spa=spa,mina=.8,maxa=1.2,Nmock=Nmock,v='',wo='sm'+flout,diro=outdir)


al = [] #list to be filled with alpha values
for i in range(0,len(lik)):
	a = .8+spa/2.+spa*i
	al.append(a)
#below assumes you have matplotlib to plot things, if not, save the above info to a file or something

sigs = sigreg_c12(al,lik)
print('result is alpha = '+str((sigs[2]+sigs[1])/2.)+'+/-'+str((sigs[2]-sigs[1])/2.))

alpha = round((sigs[2]+sigs[1])/2.,4)
err = round((sigs[2]-sigs[1])/2.,3)
plt.plot(al,lik-min(lik),'k-',label='BAO template')
plt.plot(al,liksm-min(lik),'k:',label='no BAO')
plt.xlabel(r'$\alpha$ (relative isotropic BAO scale)')
plt.ylabel(r'$\Delta\chi^{2}$')
plt.title(catver+' blinded '+tp+' '+zr+r' $\alpha=$'+str(alpha)+'$\pm$'+str(err))
plt.grid()
plt.legend()
plt.savefig(outdir+'/alpha_iso_likelihood_'+flout+'.png')
#plt.show()

plt.errorbar(rl,rl**2.*xid,rl**2*diag,fmt='o',color=color)
fmod = outdir+'ximod'+tp+zr+wm+str(bs)+'.dat'
mod = np.loadtxt(fmod).transpose()
plt.plot(mod[0],mod[0]**2.*mod[1],'k-',label=r'$\chi^2$/dof='+str(round(minchi2,3))+'/'+str(nbin-5))
plt.grid()
plt.xlabel(r'$s$ (Mpc/h)')
plt.ylabel(r'$s^2\xi_0$')
plt.title(catver+' blinded '+tp+' '+zr)
plt.legend()
plt.savefig(outdir+'/xi0_1D_modelfit_'+flout+'.png')
#plt.xlim(20,rmax+10)
#plt.ylim(-50,100)
#plt.show()

plt.errorbar(mod[0],xid[3:32]-mod[1],diag[3:32],fmt='o',color=color)
plt.grid()
plt.xlabel(r'$s$ (Mpc/h)')
plt.ylabel(r'$\xi_0-\xi_{0,{\rm mod}}$')
plt.title(catver+' blinded '+tp+' '+zr)
plt.savefig(outdir+'/xi0_1D_model_residual_'+flout+'.png')
#plt.legend()
#plt.xlim(20,rmax+10)
#plt.ylim(-50,100)
#plt.show()

print('the best-fit alpha, uncertainty, and minimumu chi2 from the 1D fit are:')
print(alpha,err,minchi2)
print('results and plot are saved in '+outdir)


