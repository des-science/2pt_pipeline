import numpy as np
import treecorr
import twopoint
import fitsio as fio
import healpy as hp
from numpy.lib.recfunctions import append_fields, rename_fields
from .stage import PipelineStage, TWO_POINT_NAMES
import os

CORES_PER_TASK=20

global_measure_2_point = None

def task(ijk):
    i,j,k=ijk
    global_measure_2_point.call_treecorr(i,j,k)

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
        self.load_data()
        self.load_metadata()

        self.source_binning = load_bin_weight_files("nz_source",self.gold['objid'])
        self.weight         = load_bin_weight_files("weight",self.gold['objid'])
        if self.params['lensfile'] != 'None':
            self.lens_binning   = load_bin_weight_files("nz_lens",self.lens['objid'])
            self.lensweight     = self.lens['weight']
            self.ran_binning    = np.load(self.input_path("randoms"))
            if len(self.ran_binning)!=len(self.randoms):
                raise ValueError('bad binning or weight in file '+self.input_path("randoms"))
        
        global global_measure_2_point
        global_measure_2_point = self

    def run(self):
        #This is a parallel job

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
        all_calcs = [(i,j,k) for i in xrange(nbin) for j in xrange(nbin) for k in xrange(3)]
        calcs=[]
        for i,j,k in all_calcs:
            if (k==0)&(i<=j)&(j<self.zbins)&(self.params['2pt_only'].lower() in [None,'shear-shear','all']):
                calcs.append((i,j,k))
            if self.params['lensfile'] != 'None':
                if (k==1)&(i<self.lens_zbins)&(j<self.zbins)&(self.params['2pt_only'].lower() in [None,'pos-shear','all']):
                    calcs.append((i,j,k))
                if (k==2)&(i<=j)&(j<self.lens_zbins)&(self.params['2pt_only'].lower() in [None,'pos-pos','all']):
                    calcs.append((i,j,k))

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

    def load_array(self,d,file):

        import time
        t0=time.time()

        if self.params['has_sheared'] & (file=='shapefile'):
            d['flags_1p'] = 'flags_select_1p'
            d['flags_1m'] = 'flags_select_1m'
            d['flags_2p'] = 'flags_select_2p'
            d['flags_2m'] = 'flags_select_2m'

        if self.params['pdf_type']=='pdf':
            keys = [key for key in d.keys() if (d[key] is not None)&(key is not 'pzstack')]
        else:
            keys = [key for key in d.keys() if (d[key] is not None)]

        if 'objid' in keys:
            dtypes = [('objid','i8')]
        else:
            raise ValueError('missing object id in '+file)
        dtypes += [(key,'f8') for key in keys if (key is not 'objid')]
        if self.params['pdf_type']=='pdf':
            dtypes  += [('pzstack_'+str(i),'f8') for i in range(len(self.params['pdf_z']))]

        fits = fio.FITS(self.params[file])[-1]
        array = fits.read(columns=[d[key] for key in keys])

        array = rename_fields(array,{v: k for k, v in d.iteritems()})

        if ('weight' not in array.dtype.names) & (file=='shapefile'):
            array = append_fields(array, 'weight', np.ones(len(array)), usemask=False)

        if self.params['pdf_type']=='pdf':
            for i in range(len(self.params['pdf_z'])):
                array['pzstack'+str(i)]=fits.read(columns=d['pzstack']+str(i))

        if np.any(np.diff(array['objid']) < 1):
            raise ValueError('misordered or duplicate ids in '+file) 

        return array


    def load_metadata(self):
        import yaml
        filename = self.input_path('nofz_meta')
        data = yaml.load(open(filename))
        self.mean_e1 = np.array(data['mean_e1'])
        self.mean_e2 = np.array(data['mean_e2'])

    def load_data(self):

        import time
        import importlib
        col = importlib.import_module('.'+self.params['dict_file'],'pipeline')

        # def lowercase_array(arr):
        #     old_names = arr.dtype.names
        #     new_names = [name.lower() for name in old_names]
        #     renames   = dict(zip(old_names, new_names))
        #     return rename_fields(arr, renames)
        t0 = time.time()

        self.gold      = self.load_array(col.gold_dict, 'goldfile')
        print 'Done goldfile',time.time()-t0,self.gold.dtype.names
        self.shape     = self.load_array(col.shape_dict, 'shapefile')
        print 'Done shapefile',time.time()-t0,self.shape.dtype.names
        self.pz        = self.load_array(col.pz_bin_dict, 'photozfile')
        print 'Done pzfile',time.time()-t0,self.pz.dtype.names
        if self.params['has_sheared']:
            self.pz_1p = self.load_array(col.pz_bin_dict, 'photozfile_1p')
            print 'Done pz1pfile',time.time()-t0,self.pz_1p.dtype.names
            self.pz_1m = self.load_array(col.pz_bin_dict, 'photozfile_1m')
            print 'Done pz1mfile',time.time()-t0,self.pz_1m.dtype.names
            self.pz_2p = self.load_array(col.pz_bin_dict, 'photozfile_2p')
            print 'Done pz2pfile',time.time()-t0,self.pz_2p.dtype.names
            self.pz_2m = self.load_array(col.pz_bin_dict, 'photozfile_2m')
            print 'Done pz2mfile',time.time()-t0,self.pz_2m.dtype.names
        self.pz_nofz   = self.load_array(col.pz_stack_dict, 'photozfile_nz')
        print 'Done pznofzfile',time.time()-t0,self.pz_nofz.dtype.names
        if self.params['lensfile'] != 'None':
            self.lens      = self.load_array(col.lens_dict, 'lensfile')
            print 'Done lensfile',time.time()-t0,self.lens.dtype.names
            self.lens_pz   = self.load_array(col.lens_pz_dict, 'lensfile')
            print 'Done lens_pzfile',time.time()-t0,self.lens_pz.dtype.names

        if 'm1' not in self.shape.dtype.names:
            self.shape = append_fields(self.shape, 'm1', self.shape['m2'], usemask=False)
        if 'm2' not in self.shape.dtype.names:
            self.shape = append_fields(self.shape, 'm2', self.shape['m1'], usemask=False)
        if self.params['oneplusm']==False:
            print 'converting m to 1+m'
            self.shape['m1'] = np.copy(self.shape['m1'])+1.
            self.shape['m2'] = np.copy(self.shape['m2'])+1.
        if 'c1' in self.shape.dtype.names:
            self.shape['e1'] -= self.shape['c1']
            self.shape['e2'] -= self.shape['c2']
            self.shape['c1'] = None
            self.shape['c2'] = None
        if self.params['flip_e2']==True:
            print 'flipping e2'
            self.shape['e2']*=-1
        if self.params['lensfile'] != 'None':
            if 'pzbin' not in self.lens_pz.dtype.names:
                self.lens_pz = append_fields(self.lens_pz, 'pzbin', self.lens_pz['pzstack'], usemask=False)
            if 'pzstack' not in self.lens_pz.dtype.names:
                self.lens_pz = append_fields(self.lens_pz, 'pzstack', self.lens_pz['pzbin'], usemask=False)

        if not ((len(self.gold)==len(self.shape))
            & (len(self.gold)==len(self.pz))
            & (len(self.gold)==len(self.pz_nofz))):
            raise ValueError('shape, gold, or photoz length mismatch')
        if self.params['has_sheared']:
            if not ((len(self.gold)==len(self.pz_1p))
                & (len(self.gold)==len(self.pz_1m))
                & (len(self.gold)==len(self.pz_2p))
                & (len(self.gold)==len(self.pz_2m))):
                raise ValueError('shape, gold, or photoz length mismatch')        
        if self.params['lensfile'] != 'None':
            if (len(self.lens)!=len(self.lens_pz)):
                raise ValueError('lens and lens_pz length mismatch') 

        if self.params['lensfile'] != 'None':
            keys = [key for key in col.ran_dict.keys() if (col.ran_dict[key] is not None)]
            fits = fio.FITS(self.params['randomfile'])[-1]

            dtypes=[(key,'f8') for key in keys]
            self.randoms = np.empty(fits.read_header()['NAXIS2'], dtype = dtypes)
            for key in keys:
                self.randoms[key]=fits.read(columns=[col.ran_dict[key]])

        if self.params['test_run']==True:
            idx = np.load(self.input_path("gold_idx"))
            self.gold    = self.gold[idx]
            self.shape   = self.shape[idx]
            self.pz      = self.pz[idx]
            self.pz_nofz = self.pz_nofz[idx]
            if self.params['has_sheared']:
                self.pz_1p   = self.pz_1p[idx]
                self.pz_1m   = self.pz_1m[idx]
                self.pz_2p   = self.pz_2p[idx]
                self.pz_2m   = self.pz_2m[idx]
            if self.params['lensfile'] != 'None':
                idx = np.load(self.input_path("lens_idx"))
                self.lens    = self.lens[idx]
                self.lens_pz = self.lens_pz[idx]
                idx = np.load(self.input_path("ran_idx"))
                self.randoms = self.randoms[idx]

        if 'pzbin_col' in self.gold.dtype.names:
            mask = (self.gold['pzbin_col'] >= 0)
        else:
            mask = (self.pz['pzbin'] > self.params['zlims'][0]) & (self.pz['pzbin'] <= self.params['zlims'][1])
            if self.params['has_sheared']:
                mask_1p = (self.pz_1p['pzbin'] > self.params['zlims'][0]) & (self.pz_1p['pzbin'] <= self.params['zlims'][1])
                mask_1m = (self.pz_1m['pzbin'] > self.params['zlims'][0]) & (self.pz_1m['pzbin'] <= self.params['zlims'][1])
                mask_2p = (self.pz_2p['pzbin'] > self.params['zlims'][0]) & (self.pz_2p['pzbin'] <= self.params['zlims'][1])
                mask_2m = (self.pz_2m['pzbin'] > self.params['zlims'][0]) & (self.pz_2m['pzbin'] <= self.params['zlims'][1])

        if 'flags' in self.shape.dtype.names:
            mask = mask & (self.shape['flags']==0)
            if self.params['has_sheared']:
                mask_1p = mask_1p & (self.shape['flags_1p']==0)
                mask_1m = mask_1m & (self.shape['flags_1m']==0)
                mask_2p = mask_2p & (self.shape['flags_2p']==0)
                mask_2m = mask_2m & (self.shape['flags_2m']==0)
        if 'flags' in self.pz.dtype.names:
            mask = mask & (self.pz['flags']==0)
            if self.params['has_sheared']:
                mask_1p = mask_1p & (self.pz_1p['flags']==0)
                mask_1m = mask_1m & (self.pz_1m['flags']==0)
                mask_2p = mask_2p & (self.pz_2p['flags']==0)
                mask_2m = mask_2m & (self.pz_2m['flags']==0)

        print 'hardcoded spt region cut'
        mask = mask & (self.shape['dec']<-35)
        if self.params['has_sheared']:
            mask_1p = mask_1p & (self.shape['dec']<-35)
            mask_1m = mask_1m & (self.shape['dec']<-35)
            mask_2p = mask_2p & (self.shape['dec']<-35)
            mask_2m = mask_2m & (self.shape['dec']<-35)

        if 'footprintfile' in self.params.keys():
            print 'cutting catalog to footprintfile'
            footmask = np.in1d(hp.ang2pix(4096, np.pi/2.-np.radians(self.shape['dec']),np.radians(self.shape['ra']), nest=False),
                               fio.FITS(self.params['footprintfile'])[-1].read()['HPIX'],assume_unique=False)
            mask = mask & footmask
            if self.params['has_sheared']:
                mask_1p = mask_1p & footmask
                mask_1m = mask_1m & footmask
                mask_2p = mask_2p & footmask
                mask_2m = mask_2m & footmask

        if self.params['has_sheared']:
            full_mask     = mask | mask_1p | mask_1m | mask_2p | mask_2m
            self.mask     = mask[full_mask]
            self.mask_1p  = mask_1p[full_mask]
            self.mask_1m  = mask_1m[full_mask]
            self.mask_2p  = mask_2p[full_mask]
            self.mask_2m  = mask_2m[full_mask]
        else:
            full_mask  = mask
            self.mask  = mask[full_mask]
        self.gold      = self.gold[full_mask]
        self.shape     = self.shape[full_mask]

        self.pz = None
        self.pz_1p = None
        self.pz_1m = None
        self.pz_2p = None
        self.pz_2m = None
        self.pz_nofz = None

        return
    

    def call_treecorr(self,i,j,k):
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
            theta,xi,xi2,xierr,xi2err,npairs,weight = self.calc_shear_shear(i,j,verbose,num_threads)
            self.xi.append([xi,xi2,None,None])
            self.xierr.append([xierr,xi2err,None,None])
        if (k==1): # gammat
            theta,xi,xierr,npairs,weight = self.calc_pos_shear(i,j,verbose,num_threads)
            self.xi.append([None,None,xi,None])
            self.xierr.append([None,None,xierr,None])
        if (k==2): # wtheta
            theta,xi,xierr,npairs,weight = self.calc_pos_pos(i,j,verbose,num_threads)
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

    def get_m(self,i):

        if self.params['has_sheared']:
            mask = (self.source_binning[0] == i)
            mask_1p = (self.source_binning[1] == i)
            mask_1m = (self.source_binning[2] == i)
            mask_2p = (self.source_binning[3] == i)
            mask_2m = (self.source_binning[4] == i)
            m1   = np.mean(self.shape['m1'][mask&self.mask])
            m2   = np.mean(self.shape['m2'][mask&self.mask])
            m1   += (np.mean(self.shape['e1'][mask_1p&self.mask_1p]) - np.mean(self.shape['e1'][mask_1m&self.mask_1m])) / (2.*self.params['dg'])
            m2   += (np.mean(self.shape['e2'][mask_2p&self.mask_2p]) - np.mean(self.shape['e2'][mask_2m&self.mask_2m])) / (2.*self.params['dg'])
            m1   = m1*np.ones(len(mask))
            m2   = m2*np.ones(len(mask))
        else:
            mask = (self.source_binning == i)
            m1 = self.shape['m1']
            m2 = self.shape['m2']

        return m1, m2, mask&self.mask

    def calc_shear_shear(self,i,j,verbose,num_threads):

        m1,m2,mask = self.get_m(i)
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

    def calc_pos_shear(self,i,j,verbose,num_threads):

        mask = self.lens_binning==i
        lenscat_i = treecorr.Catalog(w=self.lensweight[mask], ra=self.lens['ra'][mask], dec=self.lens['dec'][mask], ra_units='deg', dec_units='deg')

        mask = self.ran_binning==i
        rancat_i  = treecorr.Catalog(w=np.ones(np.sum(mask)), ra=self.randoms['ra'][mask], dec=self.randoms['dec'][mask], ra_units='deg', dec_units='deg')

        m1,m2,mask = self.get_m(j)
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

    def calc_pos_pos(self,i,j,verbose,num_threads):

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

