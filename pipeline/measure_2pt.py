import numpy as np
import treecorr
import twopoint
import h5py
import fitsio as fio
import healpy as hp
from numpy.lib.recfunctions import append_fields, rename_fields
from .stage import PipelineStage, TWO_POINT_NAMES
import os

CORES_PER_TASK=20

global_measure_2_point = None

def task(ijk):
    i,j,pix,k=ijk
    global_measure_2_point.call_treecorr(i,j,pix,k)

class Measure2Point(PipelineStage):
    name = "2pt"
    inputs = {
        "weight"        : ("nofz", "weight.npy")          ,
        "nz_source"     : ("nofz", "nz_source_zbin.npy")  ,
        "nz_lens"       : ("nofz", "nz_lens_zbin.npy")    ,
        "randoms"       : ("nofz", "randoms_zbin.npy")    ,
        "gold_idx"      : ("nofz", "gold_idx.npy")        ,
        "lens_idx"      : ("nofz", "lens_idx.npy")        ,
        "ran_idx"       : ("nofz", "ran_idx.npy")         ,
        "nofz_meta"     : ("nofz", "metadata.yaml")       ,

    }
    outputs = {
        "xip"    : "{rank}_xip.txt",
        "xim"    : "{rank}_xim.txt",
        "gammat" : "{rank}_gammat.txt",
        "wtheta" : "{rank}_wtheta.txt",
        "any"    : "*{rank}*",
    }

    def __init__(self,param_file):
        """
        Initialise object and load catalogs.
        """
        super(Measure2Point,self).__init__(param_file)

        import importlib
        col = importlib.import_module('.'+self.params['dict_file'],'pipeline')

        def load_bin_weight_files(file,objid):

            filename = self.input_path(file)
            array    = np.load(filename)

            if np.array_equal(array[:,0], objid):
                if (file == "nz_source")&self.params['has_sheared']:
                    return array[:,1],array[:,2],array[:,3],array[:,4],array[:,5]
                else:
                    return array[:,1]
            else:
                print array[:,0],len(array[:,0]),objid,len(objid)
                raise ValueError('bad binning or weight in file '+filename)

        import glob
        if (self.params['2pt_only'] == 'shear-shear')|(self.params['lensfile'] == 'None'):
            names = TWO_POINT_NAMES[:2]
        elif self.params['2pt_only'] == 'pos-shear':
            names = TWO_POINT_NAMES[2:3]
        elif self.params['2pt_only'] == 'pos-pos':
            names = TWO_POINT_NAMES[3:]
        else:
            names = TWO_POINT_NAMES
        for n,name in enumerate(names):
            files=glob.glob(self.output_path('any').format(rank=name))
            for file in files:
                try:
                    os.remove(file)
                except OSError:
                    print 'OSerror removing old 2pt file'

        # Load data
        # self.load_data()
        self.load_metadata()
                
        mcal_file = 'destest_mcal.yaml'
        params_mcal = yaml.load(open(mcal_file))
        params_mcal['param_file'] = mcal_file
        source_mcal = destest.H5Source(params_mcal)
        self.selector_mcal = destest.Selector(params_mcal,source_mcal)
        self.calibrator = destest.MetaCalib(params_mcal,self.selector_mcal)

        gold_file = 'destest_gold.yaml'
        params_gold = yaml.load(open(gold_file))
        params_gold['param_file'] = gold_file
        source_gold = destest.H5Source(params_gold)
        self.selector_gold = destest.Selector(params_gold,source_gold,inherit=self.selector_mcal)

        if self.params['flip_e2']==True:
            print 'flipping e2'
            self.shape['e2']*=-1

        self.weight         = load_bin_weight_files("weight",self.gold['objid'])
        if self.params['lensfile'] != 'None':
            self.lens_binning   = load_bin_weight_files("nz_lens",self.lens['objid'])
            self.lensweight     = self.lens['weight']
            self.ran_binning    = np.load(self.input_path("randoms"))
            if len(self.ran_binning)!=len(self.randoms):
                raise ValueError('bad binning or weight in file '+self.input_path("randoms"))
        
        global global_measure_2_point
        global_measure_2_point = self

    def get_hpix(self):

        for nside in range(1,20):
            if self.params['tbounds'][1]>hp.nside2resol(nside,arcmin=True):
                nside -=1
                break

        return nside, self.selector_gold.get_col(self.Dict.gold_dict['hpix']) // ( self.params['hpix_nside'] // nside )

    def setup_jobs(self):

        if hasattr(self.params['zbins'], "__len__"):
            self.zbins=len(self.params['zbins'])-1
        else:
            self.zbins=self.params['zbins']

        if self.params['lensfile'] != 'None':
            if hasattr(self.params['lens_zbins'], "__len__"):
                self.lens_zbins=len(self.params['lens_zbins'])-1
            else:
                self.lens_zbins=self.params['lens_zbins']

        if self.params['lensfile'] != 'None':
            nbin=max(self.lens_zbins,self.zbins)
        else:
            nbin=self.zbins

        all_calcs = [(i,j) for i in xrange(nbin) for j in xrange(nbin)]

        nside,pix = self.get_hpix()
        pix = np.unique(pix)

        calcs=[]
        for i,j in all_calcs:
            for pix_ in pix:
                if (i<=j)&(j<self.zbins)&(self.params['2pt_only'].lower() in [None,'shear-shear','all']):
                    calcs.append((i,j,pix_,0))
        if self.params['lensfile'] != 'None':
            for i,j in all_calcs:
                for pix_ in pix:
                    if (i<self.lens_zbins)&(j<self.zbins)&(self.params['2pt_only'].lower() in [None,'pos-shear','all']):
                        calcs.append((i,j,pix_,1))
            for i,j in all_calcs:
                for pix_ in pix:
                    if (i<=j)&(j<self.lens_zbins)&(self.params['2pt_only'].lower() in [None,'pos-pos','all']):
                        calcs.append((i,j,pix_,2))

        return calcs

    def run(self):
        #This is a parallel job

        calcs = self.setup_jobs()

        self.theta  = []
        self.xi     = []
        self.xierr  = []
        self.npairs = []
        self.weight_ = []
        self.calc   = []

        if self.comm:
            from .mpi_pool import MPIPool
            pool = MPIPool(self.comm)
            pool.map(task, calcs)
            pool.close()
        else:
            map(task, calcs)

    def load_metadata(self):
        import yaml
        filename = self.input_path('nofz_meta')
        data = yaml.load(open(filename))
        self.mean_e1 = np.array(data['mean_e1'])
        self.mean_e2 = np.array(data['mean_e2'])

    def call_treecorr(self,i,j,pix,k):
        """
        This is a wrapper for interaction with treecorr.
        """
        print "Running 2pt analysis on pair {},{},{}".format(i, j, k)
        # k==0: xi+-
        # k==1: gammat
        # k==2: wtheta
        
        verbose=0
        # Cori value
        num_threads=CORES_PER_TASK

        # if k!=1:
        #     return 0

        if (k==0): # xi+-
            theta,xi,xi2,xierr,xi2err,npairs,weight = self.calc_shear_shear(i,j,pix,verbose,num_threads)
            self.xi.append([xi,xi2,None,None])
            self.xierr.append([xierr,xi2err,None,None])
        if (k==1): # gammat
            theta,xi,xierr,npairs,weight = self.calc_pos_shear(i,j,pix,verbose,num_threads)
            self.xi.append([None,None,xi,None])
            self.xierr.append([None,None,xierr,None])
        if (k==2): # wtheta
            theta,xi,xierr,npairs,weight = self.calc_pos_pos(i,j,pix,verbose,num_threads)
            self.xi.append([None,None,None,xi])
            self.xierr.append([None,None,None,xierr])

        if i==j:
            npairs/=2
            weight/=2

        self.theta.append(theta)
        self.npairs.append(npairs)
        self.weight_.append(weight)
        self.calc.append((i,j,k))
        if k==0:
            np.savetxt('2pt_'+str(i)+'_'+str(j)+'_'+str(k)+'.txt',np.vstack((theta,xi,xi2,npairs,weight)).T)
        else:
            np.savetxt('2pt_'+str(i)+'_'+str(j)+'_'+str(k)+'.txt',np.vstack((theta,xi,npairs,weight)).T)

        return 0

    def get_zbins_R(self,i):

        f = h5py.File( self.input_path("nz_source"), mode='r+')
        source_binning = []
        for zbin_ in f['nofz'].keys()
            source_binning.append(f['nofz'][zbin_][:])

        if self.params['has_sheared']:
            mask = (source_binning[0] == i)
            mask_1p = (source_binning[1] == i)
            mask_1m = (source_binning[2] == i)
            mask_2p = (source_binning[3] == i)
            mask_2m = (source_binning[4] == i)

            R1,c,w = self.calibrator.calibrate('e1',mask=[mask,mask_1p,mask_1m,mask_2p,mask_2m])
            R2,c,w = self.calibrator.calibrate('e2',mask=[mask,mask_1p,mask_1m,mask_2p,mask_2m])
        else:
            mask = (self.source_binning == i)
            R1 = self.shape['m1']
            R2 = self.shape['m2']

        return R1, R2, mask, w

    def build_source_catalog(self,i,ipix,pix):

        R1,R2,mask,w = self.get_zbins_R(i)
        s = np.argsort(pix[mask])

        g1=(self.selector_mcal.get_col(self.Dict.mcal_dict['e1'])-self.mean_e1[i])/R1
        g2=(self.selector_mcal.get_col(self.Dict.mcal_dict['e1'])-self.mean_e2[i])/R2
        ra=self.selector_gold.get_col(self.Dict.gold_dict['ra'])
        dec=self.selector_gold.get_col(self.Dict.gold_dict['dec'])

        if np.isscalar(w):
            cat = treecorr.Catalog(g1=g1[mask][s], g2=g2[mask][s], ra=ra[mask][s], dec=dec[mask][s], ra_units='deg', dec_units='deg')
        else:
            cat = treecorr.Catalog(g1=g1[mask][s], g2=g2[mask][s], w=w[mask][s], ra=ra[mask][s], dec=dec[mask][s], ra_units='deg', dec_units='deg')

        return cat,pix[mask][s]

    def calc_shear_shear(self,i,j,ipix,verbose,num_threads):

        nside,pix = self.get_hpix()
        cat_i,pix = self.build_source_catalog(i,ipix,pix)

        print mask,len(mask)
        if self.params['has_sheared']:
            cat_i = treecorr.Catalog(g1=(self.shape['e1'][mask]-self.mean_e1[i])/m1[mask], g2=(self.shape['e2'][mask]-self.mean_e2[i])/m2[mask], w=self.weight[mask], ra=self.shape['ra'][mask], dec=self.shape['dec'][mask], ra_units='deg', dec_units='deg')
        else:
            cat_i = treecorr.Catalog(g1=(self.shape['e1'][mask]-self.mean_e1[i]), g2=(self.shape['e2'][mask]-self.mean_e2[i]), w=self.weight[mask], ra=self.shape['ra'][mask], dec=self.shape['dec'][mask], ra_units='deg', dec_units='deg')
            biascat_i = treecorr.Catalog(k=np.sqrt(self.shape['m1'][mask]*self.shape['m2'][mask]), w=self.weight[mask], ra=self.shape['ra'][mask], dec=self.shape['dec'][mask], ra_units='deg', dec_units='deg')

        m1,m2,mask = self.get_m(j)
        if self.params['has_sheared']:
            cat_j = treecorr.Catalog(g1=(self.shape['e1'][mask]-self.mean_e1[j])/m1[mask], g2=(self.shape['e2'][mask]-self.mean_e2[j])/m2[mask], w=self.weight[mask], ra=self.shape['ra'][mask], dec=self.shape['dec'][mask], ra_units='deg', dec_units='deg')
        else:
            cat_j = treecorr.Catalog(g1=(self.shape['e1'][mask]-self.mean_e1[j]), g2=(self.shape['e2'][mask]-self.mean_e2[j]), w=self.weight[mask], ra=self.shape['ra'][mask], dec=self.shape['dec'][mask], ra_units='deg', dec_units='deg')
            biascat_j = treecorr.Catalog(k=np.sqrt(self.shape['m1'][mask]*self.shape['m2'][mask]), w=self.weight[mask], ra=self.shape['ra'][mask], dec=self.shape['dec'][mask], ra_units='deg', dec_units='deg')

        gg = treecorr.GGCorrelation(nbins=self.params['tbins'], min_sep=self.params['tbounds'][0], max_sep=self.params['tbounds'][1], sep_units='arcmin', bin_slop=self.params['slop'], verbose=verbose,num_threads=num_threads)
        gg.process(cat_i,cat_j)
        if self.params['has_sheared']:
            norm = 1.
        else:
            kk = treecorr.KKCorrelation(nbins=self.params['tbins'], min_sep=self.params['tbounds'][0], max_sep=self.params['tbounds'][1], sep_units='arcmin', bin_slop=self.params['slop'], verbose=verbose,num_threads=num_threads)
            kk.process(biascat_i,biascat_j)
            norm = kk.xi

        theta=np.exp(gg.meanlogr)
        xip = gg.xip/norm
        xim = gg.xim/norm
        xiperr = ximerr = np.sqrt(gg.varxi)/norm

        return theta, xip, xim, xiperr, ximerr, gg.npairs, gg.weight

    def calc_pos_shear(self,i,j,pix,verbose,num_threads):

        mask = self.lens_binning==i
        lenscat_i = treecorr.Catalog(w=self.lensweight[mask], ra=self.lens['ra'][mask], dec=self.lens['dec'][mask], ra_units='deg', dec_units='deg')

        mask = self.ran_binning==i
        rancat_i  = treecorr.Catalog(w=np.ones(np.sum(mask)), ra=self.randoms['ra'][mask], dec=self.randoms['dec'][mask], ra_units='deg', dec_units='deg')

        m1,m2,mask = self.get_zbins_R(j)
        print mask,len(mask)
        if self.params['has_sheared']:
            cat_j = treecorr.Catalog(g1=(self.shape['e1'][mask]-self.mean_e1[j])/m1[mask], g2=(self.shape['e2'][mask]-self.mean_e2[j])/m2[mask], w=self.weight[mask], ra=self.shape['ra'][mask], dec=self.shape['dec'][mask], ra_units='deg', dec_units='deg')
        else:
            print 'e1',self.shape['e1'][mask]
            print 'me1',j,self.mean_e1[j],j
            print 'weight',self.weight[mask]
            print 'm1',self.shape['m1'][mask]
            cat_j = treecorr.Catalog(g1=(self.shape['e1'][mask]-self.mean_e1[j]), g2=(self.shape['e2'][mask]-self.mean_e2[j]), w=self.weight[mask], ra=self.shape['ra'][mask], dec=self.shape['dec'][mask], ra_units='deg', dec_units='deg')
            biascat_j = treecorr.Catalog(k=np.sqrt(self.shape['m1'][mask]*self.shape['m2'][mask]), w=self.weight[mask], ra=self.shape['ra'][mask], dec=self.shape['dec'][mask], ra_units='deg', dec_units='deg')

        ng = treecorr.NGCorrelation(nbins=self.params['tbins'], min_sep=self.params['tbounds'][0], max_sep=self.params['tbounds'][1], sep_units='arcmin', bin_slop=self.params['slop'], verbose=verbose,num_threads=num_threads)
        rg = treecorr.NGCorrelation(nbins=self.params['tbins'], min_sep=self.params['tbounds'][0], max_sep=self.params['tbounds'][1], sep_units='arcmin', bin_slop=self.params['slop'], verbose=verbose,num_threads=num_threads)
        if self.params['has_sheared']:
            norm = 1.
        else:
            nk = treecorr.NKCorrelation(nbins=self.params['tbins'], min_sep=self.params['tbounds'][0], max_sep=self.params['tbounds'][1], sep_units='arcmin', bin_slop=self.params['slop'], verbose=verbose,num_threads=num_threads)
            nk.process(lenscat_i,biascat_j)
            norm,tmp=nk.calculateXi()
        ng.process(lenscat_i,cat_j)
        rg.process(rancat_i,cat_j)
        gammat,gammat_im,gammaterr=ng.calculateXi(rg)

        theta=np.exp(ng.meanlogr)
        if np.sum(norm)==0:
          norm=1.
        gammat/=norm
        gammat_im/=norm
        gammaterr=np.sqrt(gammaterr/norm)

        return theta, gammat, gammaterr, ng.npairs, ng.weight

    def calc_pos_pos(self,i,j,pix,verbose,num_threads):

        mask = self.lens_binning==i
        lenscat_i = treecorr.Catalog(w=self.lensweight[mask], ra=self.lens['ra'][mask], dec=self.lens['dec'][mask], ra_units='deg', dec_units='deg')

        mask = self.ran_binning==i
        rancat_i  = treecorr.Catalog(w=np.ones(np.sum(mask)), ra=self.randoms['ra'][mask], dec=self.randoms['dec'][mask], ra_units='deg', dec_units='deg')

        mask = self.lens_binning==j
        lenscat_j = treecorr.Catalog(w=self.lensweight[mask], ra=self.lens['ra'][mask], dec=self.lens['dec'][mask], ra_units='deg', dec_units='deg')

        mask = self.ran_binning==j
        rancat_j  = treecorr.Catalog(w=np.ones(np.sum(mask)), ra=self.randoms['ra'][mask], dec=self.randoms['dec'][mask], ra_units='deg', dec_units='deg')

        nn = treecorr.NNCorrelation(nbins=self.params['tbins'], min_sep=self.params['tbounds'][0], max_sep=self.params['tbounds'][1], sep_units='arcmin', bin_slop=self.params['slop'], verbose=verbose,num_threads=num_threads)
        rn = treecorr.NNCorrelation(nbins=self.params['tbins'], min_sep=self.params['tbounds'][0], max_sep=self.params['tbounds'][1], sep_units='arcmin', bin_slop=self.params['slop'], verbose=verbose,num_threads=num_threads)
        nr = treecorr.NNCorrelation(nbins=self.params['tbins'], min_sep=self.params['tbounds'][0], max_sep=self.params['tbounds'][1], sep_units='arcmin', bin_slop=self.params['slop'], verbose=verbose,num_threads=num_threads)
        rr = treecorr.NNCorrelation(nbins=self.params['tbins'], min_sep=self.params['tbounds'][0], max_sep=self.params['tbounds'][1], sep_units='arcmin', bin_slop=self.params['slop'], verbose=verbose,num_threads=num_threads)
        nn.process(lenscat_i,lenscat_j)
        rn.process(rancat_i,lenscat_j)
        nr.process(lenscat_i,rancat_j)
        rr.process(rancat_i,rancat_j)

        theta=np.exp(nn.meanlogr)
        wtheta,wthetaerr=nn.calculateXi(rr,dr=nr,rd=rn)
        wthetaerr=np.sqrt(wthetaerr)

        return theta, wtheta, wthetaerr, nn.npairs, nn.weight

    def write(self):
        """
        Write data to files.
        """
        if self.comm is None:
            rank = 0
        else:
            rank = self.comm.Get_rank()

        for n,name in enumerate(TWO_POINT_NAMES):
            if (n<2)&(self.params['2pt_only'].lower() not in [None,'shear-shear','all']):
                continue
            if (n==2)&((self.params['2pt_only'].lower() not in [None,'pos-shear','all'])|(self.params['lensfile'] == 'None')):
                continue
            if (n==3)&((self.params['2pt_only'].lower() not in [None,'pos-pos','all'])|(self.params['lensfile'] == 'None')):
                continue
            filename = self.output_path(name).format(rank=rank)
            f = None
            for (theta,xi_data,npairs,weight,ijk) in zip(self.theta, self.xi, self.npairs, self.weight_, self.calc):
                i,j,k = ijk
                if (n<2) and (k!=0):
                    continue
                if (n==2) and (k!=1):
                    continue
                if (n==3) and (k!=2):
                    continue
                if f is None:
                    f = open(filename, 'w')
                for theta,xi_theta,npairs_,weight_ in zip(theta, xi_data[n], npairs, weight):
                    f.write("{} {} {} {} {} {}\n".format(theta, i, j, xi_theta, npairs_, weight_))
            if f is not None:
                f.close()

