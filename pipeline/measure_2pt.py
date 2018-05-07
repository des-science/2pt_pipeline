import numpy as np
import treecorr
import twopoint
import h5py
import fitsio as fio
import healpy as hp
from numpy.lib.recfunctions import append_fields, rename_fields
from .stage import PipelineStage, TWO_POINT_NAMES
import os
import sys
import yaml
import destest
import mpi4py.MPI

CORES_PER_TASK=20

global_measure_2_point = None

def task(ijk):
    i,j,pix,k=ijk
    global_measure_2_point.call_treecorr(i,j,pix,k)

class Measure2Point(PipelineStage):
    name = "2pt"
    inputs = {
        "weight"        : ("nofz", "weight.npy")          ,
        "nz_source"     : ("nofz", "nz_source_zbin.h5")  ,
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
        self.source_selector = destest.Selector(params_mcal,source_mcal)
        self.source_calibrator = destest.MetaCalib(params_mcal,self.source_selector)

        gold_file = 'destest_gold.yaml'
        params_gold = yaml.load(open(gold_file))
        params_gold['param_file'] = gold_file
        source_gold = destest.H5Source(params_gold)
        self.gold_selector = destest.Selector(params_gold,source_gold,inherit=self.source_selector)

        lens_file = 'destest_redmagic.yaml'
        params_lens = yaml.load(open(lens_file))
        params_lens['param_file'] = lens_file
        source_lens = destest.H5Source(params_lens)
        self.lens_selector = destest.Selector(params_lens,source_lens)
        self.lens_calibrator = destest.NoCalib(params_lens,self.lens_selector)
        

        random_file = 'destest_random.yaml'
        params_random = yaml.load(open(random_file))
        params_random['param_file'] = random_file
        source_random = destest.H5Source(params_random)
        self.ran_selector = destest.Selector(params_random,source_random)

        self.Dict = importlib.import_module('.'+self.params['dict_file'],'pipeline')
        print 'using dictionary: ',self.params['dict_file']

        print 'pz selector'
        pz_file = 'destest_pz.yaml'
        params_pz = yaml.load(open(pz_file))
        params_pz['param_file'] = pz_file
        if params_pz['group'][-3:] == 'bpz':
            print "will use BPZ's column names"
            self.Dict.pz_dict = self.Dict.bpz_dict
        else:
            print "will use DNF's column names"
            self.Dict.pz_dict = self.Dict.dnf_dict
        source_pz = destest.H5Source(params_pz)
        self.selector_pz = destest.Selector(params_pz,source_pz,inherit=self.source_selector)

        
        # if self.params['flip_e2']==True:
        #     print 'flipping e2'
        #     self.shape['e2']*=-1

        #self.weight         = load_bin_weight_files("weight",self.gold['objid'])
        #if self.params['lensfile'] != 'None':
        #    self.lens_binning   = load_bin_weight_files("nz_lens",self.lens['objid'])
        #    self.lensweight     = self.lens['weight']
        #    self.ran_binning    = np.load(self.input_path("randoms"))
        #    if len(self.ran_binning)!=len(self.randoms):
        #        raise ValueError('bad binning or weight in file '+self.input_path("randoms"))
        
        global global_measure_2_point
        global_measure_2_point = self

    def get_nside(self):
        for nside in range(1,20):
            if self.params['tbounds'][1]>hp.nside2resol(2**nside,arcmin=True):
                nside -=1
                break
        return 2**nside

    
    def get_hpix(self,pix=None):
        if pix == None:
            return self.gold_selector.get_col(self.Dict.gold_dict['hpix'])[0] // ( hp.nside2npix(self.params['hpix_nside']) // hp.nside2npix(self.get_nside()) )
        else:
            return pix // ( self.params['hpix_nside'] // self.get_nside() )

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
        
        pix = self.get_hpix()
        pix = np.unique(pix)
        print '------------- number of pixels',len(pix)

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

        f = h5py.File('2pt.h5',mode='w', driver='mpio', comm=mpi4py.MPI.COMM_WORLD)
        for i,j,ipix,calc in calcs:
            for jpix in range(9):
                for d in ['meanlogr','d1','d2','npairs','weight']:
                    if calc==0:
                        f.create_dataset( '2pt/xip/'+str(ipix)+'/'+str(jpix)+'/'+str(i)+'/'+str(j)+'/'+d+'/', shape=(self.params['tbins'],), dtype=float )
                        f.create_dataset( '2pt/xim/'+str(ipix)+'/'+str(jpix)+'/'+str(i)+'/'+str(j)+'/'+d+'/', shape=(self.params['tbins'],), dtype=float )
                    if calc==1:
                        f.create_dataset( '2pt/gammat/'+str(ipix)+'/'+str(jpix)+'/'+str(i)+'/'+str(j)+'/'+d+'/', shape=(self.params['tbins'],), dtype=float )
                    if calc==2:
                        f.create_dataset( '2pt/wtheta/'+str(ipix)+'/'+str(jpix)+'/'+str(i)+'/'+str(j)+'/'+d+'/', shape=(self.params['tbins'],), dtype=float )
                for d in ['npairs','weight']:
                    if calc==2:
                        f.create_dataset( '2pt/random/'+str(ipix)+'/'+str(jpix)+'/'+str(i)+'/'+str(j)+'/'+d+'/', shape=(self.params['tbins'],), dtype=float )
        f.close()

        return calcs

    def run(self):
        #This is a parallel job

        if self.comm:
            from .mpi_pool import MPIPool
            pool = MPIPool(self.comm)
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
            calcs = self.setup_jobs()
            pool.map(task, calcs)
            pool.close()
        else:
            calcs = self.setup_jobs()
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

        if (k==0): # xi+-
            out = self.calc_shear_shear(i,j,pix,verbose,num_threads)
        if (k==1): # gammat
            out = self.calc_pos_shear(i,j,pix,verbose,num_threads)
        if (k==2): # wtheta
            out = self.calc_pos_pos(i,j,pix,verbose,num_threads)

        f = h5py.File('2pt.h5',mode='r+', driver='mpio', comm=self.comm)
        for jp in range(9):
            for di,d in tuple(zip([0,1,2],['meanlogr','d1','d2'])):
                if k==0:
                    print 'Writing in: 2pt/xip/'+str(pix)+'/'+str(jp)+'/'+str(i)+'/'+str(j)+'/'+d+'/'
                    f['2pt/xip/'+str(pix)+'/'+str(jp)+'/'+str(i)+'/'+str(j)+'/'+d+'/'][:] = out[jp,di,:]
                    f['2pt/xim/'+str(pix)+'/'+str(jp)+'/'+str(i)+'/'+str(j)+'/'+d+'/'][:] = out[jp,di,:]
                if k==1:
                    f['2pt/gammat/'+str(pix)+'/'+str(jp)+'/'+str(i)+'/'+str(j)+'/'+d+'/'][:] = out[jp,di,:]
                if k==2:
                    f['2pt/wtheta/'+str(pix)+'/'+str(jp)+'/'+str(i)+'/'+str(j)+'/'+d+'/'][:] = out[jp,di,:]
            for di,d in tuple(zip([3,4],['npairs','weight'])):
                if k==0:
                    if i==j:
                        f['2pt/xip/'+str(pix)+'/'+str(jp)+'/'+str(i)+'/'+str(j)+'/'+d+'/'][:] = out[jp,di,:]/2
                        f['2pt/xim/'+str(pix)+'/'+str(jp)+'/'+str(i)+'/'+str(j)+'/'+d+'/'][:] = out[jp,di,:]/2
                    else:
                        f['2pt/xip/'+str(pix)+'/'+str(jp)+'/'+str(i)+'/'+str(j)+'/'+d+'/'][:] = out[jp,di,:]
                        f['2pt/xim/'+str(pix)+'/'+str(jp)+'/'+str(i)+'/'+str(j)+'/'+d+'/'][:] = out[jp,di,:]
                if k==1:
                    if i==j:
                        f['2pt/gammat/'+str(pix)+'/'+str(jp)+'/'+str(i)+'/'+str(j)+'/'+d+'/'][:] = out[jp,di,:]/2
                    else:
                        f['2pt/gammat/'+str(pix)+'/'+str(jp)+'/'+str(i)+'/'+str(j)+'/'+d+'/'][:] = out[jp,di,:]
                if k==2:
                    if i==j:
                        f['2pt/wtheta/'+str(pix)+'/'+str(jp)+'/'+str(i)+'/'+str(j)+'/'+d+'/'][:] = out[jp,di,:]/2
                        f['2pt/random/'+str(pix)+'/'+str(jp)+'/'+str(i)+'/'+str(j)+'/'+d+'/'][:] = out[jp,di,:]/2
                    else:
                        f['2pt/wtheta/'+str(pix)+'/'+str(jp)+'/'+str(i)+'/'+str(j)+'/'+d+'/'][:] = out[jp,di,:]
                        f['2pt/random/'+str(pix)+'/'+str(jp)+'/'+str(i)+'/'+str(j)+'/'+d+'/'][:] = out[jp,di,:]

        f.close()

        return 0

    def get_zbins_R(self,i,cal,shape=True):

        print 'in zbins R'
        f = h5py.File( self.input_path("nz_source"), mode='r')
        if type(cal)==destest.NoCalib: # lens catalog so get random mask
            source_binning = [f['nofz/lens_zbin']]
        else:
            source_binning = []
            print 'fix to be actual number of bins'
            for zbin_ in ['zbin','zbin_1p','zbin_1m','zbin_2p','zbin_2m']:
                source_binning.append(f['nofz'][zbin_][:])
                print 'souce_binning_length',i,len(source_binning[-1])

        print 'source binning',source_binning,np.max(source_binning[0]),np.min(source_binning[0])
        mask = []
        for s in source_binning:
            mask.append( s == i )
            print 'mask length',len(mask[-1])

        print 'before pzbin ---------------------'
        pzbin = self.selector_pz.get_col(self.Dict.pz_dict['pzbin'])
        print 'before e1 ---------------------'
        pzbin = self.source_selector.get_col('e1')

        R1,c,w = cal.calibrate('e1',mask=mask)
        R2,c,w = cal.calibrate('e2',mask=mask)
        print 'calibrate done'

        if type(cal)==destest.NoCalib: # lens catalog so get random mask
            f = h5py.File( self.input_path("nz_source"), mode='r')
            rmask = f['nofz']['ran_zbin'][:] == i
            return R1, R2, mask[0], w, rmask
        else:
            return R1, R2, mask[0], w

    def build_catalogs(self,cal,i,ipix,pix,return_neighbor=False):

        def get_pix_subset(ipix,pix_,return_neighbor):

            theta,phi = hp.pix2ang(self.get_nside(),ipix,nest=True)
            jpix = hp.get_all_neighbours(self.get_nside(),theta,phi,nest=True)
            jpix = np.append(ipix,jpix)

            s = np.argsort(pix_)
            pix_ = pix_[s]
            if return_neighbor:
                pixrange = []
                pixrange2 = [0]
                tmp = 0
                for x,jp in enumerate(jpix):
                    pixrange = np.append(pixrange,np.r_[int(np.searchsorted(pix_, jp)) : int(np.searchsorted(pix_, jp, side='right'))])
                    tmp2 = np.searchsorted(pix_, jp, side='right') - np.searchsorted(pix_, jp)
                    pixrange2.append( np.s_[ int(tmp) : int(tmp + tmp2) ] )
                    tmp += tmp2
                pixrange = pixrange.astype(int)
            else:
                pixrange = np.r_[int(np.searchsorted(pix_, ipix)) : int(np.searchsorted(pix_, ipix, side='right'))]
                pixrange2 = None

            return s,pixrange,pixrange2

        print 'start build',i
        ra=self.gold_selector.get_col(self.Dict.gold_dict['ra'])[0]
        dec=self.gold_selector.get_col(self.Dict.gold_dict['dec'])[0]

        if type(cal)==destest.NoCalib: # lens catalog

            print 'nocalib'
            R1,R2,mask,w,rmask = self.get_zbins_R(i,cal,shape=False)
            s,pixrange,pixrange2 = get_pix_subset(ipix,pix[gmask][mask],return_neighbor)

            gmask = cal.selector.get_match()
            if len(ra[gmask][mask][s][pixrange])>0:
                cat = treecorr.Catalog(ra=ra[gmask][mask][s][pixrange], dec=dec[gmask][mask][s][pixrange], 
                                    ra_units='deg', dec_units='deg')
            else:
                cat = None

            ra = self.ran_selector.get_col(self.Dict.ran_dict['ra'])[0][rmask]
            dec = self.ran_selector.get_col(self.Dict.ran_dict['dec'])[0][rmask]
            pix = self.get_hpix(pix=hp.ang2pix(self.params['hpix_nside'],np.pi/2.-np.radians(dec),np.radians(ra),nest=True))
            s,pixrange,rpixrange2 = get_pix_subset(ipix,pix[rmask],return_neighbor)
            if len(ra[s][pixrange])>0:
                rcat = treecorr.Catalog(ra=ra[s][pixrange], 
                                        dec=dec[s][pixrange], 
                                        ra_units='deg', dec_units='deg')
            else:
                rcat = None
                return cat,rcat,pixrange2,rpixrange2

        else: # shape catalog

            print 'calib'
            R1,R2,mask,w = self.get_zbins_R(i,cal)
            print 'got z bins'
            s,pixrange,pixrange2 = get_pix_subset(ipix,pix[mask],return_neighbor)
            print 'got pix subset'

            print mask,s,pixrange
            print len(cal.selector.get_col(self.Dict.shape_dict['e1'])[0]),len(mask),len(s),len(pixrange)
            g1=cal.selector.get_col(self.Dict.shape_dict['e1'])[0][mask][s][pixrange]
            g1 = (g1-self.mean_e1[i])/R1
            g2=cal.selector.get_col(self.Dict.shape_dict['e2'])[0][mask][s][pixrange]
            g2 = (g2-self.mean_e2[i])/R2
            if len(g1)>0:
                cat = treecorr.Catalog(g1=g1, g2=g2, ra=ra[mask][s][pixrange], 
                                        dec=dec[mask][s][pixrange], ra_units='deg', dec_units='deg')
            else:
                cat = None
                return cat,pixrange2

        if not np.isscalar(w):
            cat.w = w

        if type(cal)==destest.NoCalib:
            return cat,rcat,pixrange2,rpixrange2

        return cat,pixrange2

    def calc_shear_shear(self,i,j,ipix,verbose,num_threads):

        print 'in shear_shear'

        pix = self.get_hpix()
        print 'before build'
        icat,pixrange = self.build_catalogs(self.source_calibrator,i,ipix,pix)
        print 'before build j'
        jcat,pixrange = self.build_catalogs(self.source_calibrator,j,ipix,pix,return_neighbor=True)

        print 'success build'
        out = np.zeros((9,7,self.params['tbins']))
        if (icat is None) or (jcat is None):
            return out
        for x in range(9):
            jcat.wpos[:]=0.
            jcat.wpos[pixrange[x]] = 1.
            gg = treecorr.GGCorrelation(nbins=self.params['tbins'], min_sep=self.params['tbounds'][0], max_sep=self.params['tbounds'][1], sep_units='arcmin', bin_slop=self.params['slop'], verbose=verbose,num_threads=num_threads)
            gg.process(icat,jcat)
            out[x,0,:] = gg.meanlogr
            out[x,1,:] = gg.xip
            out[x,2,:] = gg.xim
            out[x,3,:] = gg.npairs
            out[x,4,:] = gg.weight

        return out

    def calc_pos_shear(self,i,j,pix,verbose,num_threads):
        print 'in pos_shear'

        pix = self.get_hpix()
        icat,ircat,pixrange,rpixrange = self.build_catalogs(self.source_calibrator,i,ipix,pix)
        jcat,pixrange = self.build_catalogs(self.lens_calibrator,j,ipix,pix,return_neighbor=True)

        out = np.zeros((9,7,self.params['tbins']))
        if (icat is None) or (jcat is None):
            return out
        for x in range(9):
            jcat.wpos[:]=0.
            jcat.wpos[pixrange[x]] = 1.

            ng = treecorr.NGCorrelation(nbins=self.params['tbins'], min_sep=self.params['tbounds'][0], max_sep=self.params['tbounds'][1], sep_units='arcmin', bin_slop=self.params['slop'], verbose=verbose,num_threads=num_threads)
            rg = treecorr.NGCorrelation(nbins=self.params['tbins'], min_sep=self.params['tbounds'][0], max_sep=self.params['tbounds'][1], sep_units='arcmin', bin_slop=self.params['slop'], verbose=verbose,num_threads=num_threads)
            ng.process(icat,jcat)
            rg.process(ircat,jcat)
            gammat,gammat_im,gammaterr=ng.calculateXi(rg)
            out[x,0,:] = ng.meanlogr
            out[x,1,:] = gammat
            out[x,2,:] = gammat_im
            out[x,3,:] = gg.npairs
            out[x,4,:] = gg.weight

        return out

    def calc_pos_pos(self,i,j,pix,verbose,num_threads):
        print 'in pos_pos'

        pix = self.get_hpix()
        icat,ircat,pixrange,rpixrange = self.build_catalogs(self.lens_calibrator,i,ipix,pix)
        jcat,jrcat,pixrange,rpixrange = self.build_catalogs(self.lens_calibrator,i,ipix,pix,return_neighbor=True)

        out = np.zeros((9,7,self.params['tbins']))
        if (icat is None) or (jcat is None):
            return out
        for x in range(9):
            jcat.wpos[:]=0.
            jcat.wpos[pixrange[x]] = 1.

            nn = treecorr.NNCorrelation(nbins=self.params['tbins'], min_sep=self.params['tbounds'][0], max_sep=self.params['tbounds'][1], sep_units='arcmin', bin_slop=self.params['slop'], verbose=verbose,num_threads=num_threads)
            rn = treecorr.NNCorrelation(nbins=self.params['tbins'], min_sep=self.params['tbounds'][0], max_sep=self.params['tbounds'][1], sep_units='arcmin', bin_slop=self.params['slop'], verbose=verbose,num_threads=num_threads)
            nr = treecorr.NNCorrelation(nbins=self.params['tbins'], min_sep=self.params['tbounds'][0], max_sep=self.params['tbounds'][1], sep_units='arcmin', bin_slop=self.params['slop'], verbose=verbose,num_threads=num_threads)
            rr = treecorr.NNCorrelation(nbins=self.params['tbins'], min_sep=self.params['tbounds'][0], max_sep=self.params['tbounds'][1], sep_units='arcmin', bin_slop=self.params['slop'], verbose=verbose,num_threads=num_threads)

            nn.process(icat,jcat)
            rn.process(ircat,jcat)
            nr.process(icat,jrcat)
            rr.process(ircat,jrcat)

            wtheta,wthetaerr=nn.calculateXi(rr,dr=nr,rd=rn)
            wthetaerr=np.sqrt(wthetaerr)
            out[x,0,:] = ng.meanlogr
            out[x,1,:] = wtheta
            out[x,3,:] = nn.npairs
            out[x,4,:] = nn.weight
            out[x,5,:] = rr.npairs
            out[x,6,:] = rr.weight

        return out

    def write(self):
        """
        Write data to files.
        """

        return
