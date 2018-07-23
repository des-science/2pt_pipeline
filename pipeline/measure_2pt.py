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
import importlib
import glob

global_measure_2_point = None

def load_catalog(filename, inherit=None, return_calibrator=None):
    """
    Loads data access and calibration classes from destest for a given yaml setup file.
    """
    # Input yaml file defining catalog
    params = yaml.load(open(filename))
    params['param_file'] = filename
    # Load destest source class to manage access to file
    source = destest.H5Source(params)
    # Load destest selector class to manage access to data in a structured way
    if inherit is None:
        sel = destest.Selector(params,source)
    else:
        sel = destest.Selector(params,source,inherit=inherit)
    # Load destest calibrator class to manage calibration of the catalog
    if return_calibrator is not None:
        cal = return_calibrator(params,sel)
        return sel, cal
    else:
        return sel

def task(ijk):
    i,j,pix,k=ijk
    global_measure_2_point.call_treecorr(i,j,pix,k)

def task_cleanup(tmp):
    global_measure_2_point.cleanup(tmp)

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

    def __init__(self, param_file):
        """
        Initialise object and load catalogs.
        """
        super(Measure2Point,self).__init__(param_file)

        # Load metadata (needed for mean shear)
        self.load_metadata()

        # Default value for random subsampling factor
        if 'ran_factor' not in self.params:
            self.params['ran_factor'] = 999

        # A dictionary to homogenize names of columns in the hdf5 master catalog 
        self.Dict = importlib.import_module('.'+self.params['dict_file'],'pipeline')
        print 'using dictionary: ',self.params['dict_file']
                
        # Load data and calibration classes
        self.source_selector, self.source_calibrator = load_catalog(self.params['shape_yaml'], return_calibrator=destest.MetaCalib)
        self.lens_selector, self.lens_calibrator     = load_catalog(self.params['lens_yaml'], return_calibrator=destest.NoCalib)
        self.gold_selector = load_catalog(self.params['gold_yaml'], inherit=self.source_selector)
        self.ran_selector  = load_catalog(self.params['random_yaml'])

        self.Dict.ind = self.Dict.index_dict #a dictionary that takes unsheared,sheared_1p/1m/2p/2m as u-1-2-3-4 to deal with tuples of values returned by get_col()
        
        global global_measure_2_point
        global_measure_2_point = self

    def load_metadata(self):
        """
        Read metadata from nofz stage.
        """

        filename = self.input_path('nofz_meta')
        data = yaml.load(open(filename))
        self.mean_e1 = np.array(data['mean_e1'])
        self.mean_e2 = np.array(data['mean_e2'])
        self.zbins = len(np.array(data['source_bins'])) - 1
        self.lens_zbins = len(np.array(data['lens_bins'])) - 1

    def get_nside(self):
        """
        Get the necessary nside to prevent the largest separation of pairs from spanning more than an adjacent pair of healpixels.
        """
        # Loop over nside until the resolution is more than the largest theta separation requested
        for nside in range(1,20):
            if self.params['tbounds'][1] > hp.nside2resol(2**nside, arcmin=True):
                nside -=1
                break
        return 2**nside

    def get_hpix(self, pix=None):
        """
        Get downsampled healpix index for a given pixel, or if pix is None, return all downsampled healpix indices.
        """

        if pix is None:

            return self.gold_selector.get_col(self.Dict.gold_dict['hpix'])[self.Dict.ind['u']] // ( hp.nside2npix(self.params['hpix_nside']) // hp.nside2npix(self.get_nside()) )

        else:

            return pix // ( hp.nside2npix(self.params['hpix_nside']) // hp.nside2npix(self.get_nside()) )

    def get_lhpix(self):
        """
        Same as get_hpix(), but uses the unmaked gold catalog for matching to lenses.
        """

        return self.gold_selector.source.read(self.Dict.gold_dict['hpix'])[self.Dict.ind['u']] // ( hp.nside2npix(self.params['hpix_nside']) // hp.nside2npix(self.get_nside()) )


    def setup_jobs(self,pool):
        """
        Set up the list of jobs that must be distributed across nodes.
        """

        # Get max number of tomographic bins between lenses and sources
        if self.params['lens_yaml'] != 'None':
            nbin=max(self.lens_zbins,self.zbins)
        else:
            nbin=self.zbins

        print 'setting up ',nbin,'tomographic bins'

        # Create dummy list of pairs of tomographic bins up to nbin (max of lenses and sources)
        all_calcs = [(i,j) for i in xrange(nbin) for j in xrange(nbin)]
        
        # Get healpix values for each object and unique healpix cells
        pix = self.get_hpix()
        pix = np.unique(pix)
        print '------------- number of pixels',len(pix),pix

        # Loop over tomographic bin pairs and add to final calculation list each requested correlation and unique healpix cell
        calcs=[]
        # Loop over needed shear-shear correlations
        for i,j in all_calcs:
            # Loop over tomographic bin pairs
            for pix_ in pix:
                # Loop over unique healpix cells
                if (i<=j) & (j<self.zbins) & (self.params['2pt_only'].lower() in [None,'shear-shear','all']):
                    calcs.append((i,j,pix_,0)) # Only add to list if calculating shear-shear and a valid tomographic pair (doesn't duplicate identical i<j and i>j)

        if self.params['lens_yaml'] != 'None':
            for i,j in all_calcs:
                for pix_ in pix:
                    if (i<self.lens_zbins)&(j<self.zbins)&(self.params['2pt_only'].lower() in [None,'pos-shear','all']):
                        calcs.append((i,j,pix_,1))
            for i,j in all_calcs:
                for pix_ in pix:
                    if (i<=j)&(j<self.lens_zbins)&(self.params['2pt_only'].lower() in [None,'pos-pos','all']):
                        calcs.append((i,j,pix_,2))
        # if pool is not None:
        #     if not pool.is_master():
        #         print 'done calcs'
        #         return calcs

        # Pre-format output h5 file to contain all the necessary paths based on the final calculation list
        if pool is None:
            self.rank=0
            self.size=1
            f = h5py.File('2pt_'+str(self.rank)+'.h5',mode='w')
        else:
            self.rank = pool.rank
            self.size = pool.size
            f = h5py.File('2pt_'+str(self.rank)+'.h5',mode='w')#, driver='mpio', comm=self.comm)
        for i,j,ipix,calc in calcs[self.rank::self.size]:
                for jpix in range(9): # There will only ever be 9 pixel pair correlations - the auto-correlation and 8 neighbors
                    if calc==0:
                        for d in ['meanlogr','xip','xim','npairs','weight']:
                            f.create_dataset( '2pt/xipm/'+str(ipix)+'/'+str(jpix)+'/'+str(i)+'/'+str(j)+'/'+d, shape=(self.params['tbins'],), dtype=float )
                        f.create_dataset( '2pt/xipm/'+str(ipix)+'/'+str(jpix)+'/'+str(i)+'/'+str(j)+'/tot', shape=(1,), dtype=float )                            
                    if calc==1:
                        for d in ['meanlogr','ngxi','ngxim','rgxi','rgxim','ngnpairs','ngweight','rgnpairs','rgweight']:
                            f.create_dataset( '2pt/gammat/'+str(ipix)+'/'+str(jpix)+'/'+str(i)+'/'+str(j)+'/'+d, shape=(self.params['tbins'],), dtype=float )
                        f.create_dataset( '2pt/gammat/'+str(ipix)+'/'+str(jpix)+'/'+str(i)+'/'+str(j)+'/ngtot', shape=(1,), dtype=float )                            
                        f.create_dataset( '2pt/gammat/'+str(ipix)+'/'+str(jpix)+'/'+str(i)+'/'+str(j)+'/rgtot', shape=(1,), dtype=float )                            
                    if calc==2:
                        for d in ['meanlogr','nnnpairs','nnweight','nrnpairs','nrweight','rnnpairs','rnweight','rrnpairs','rrweight']:
                            f.create_dataset( '2pt/wtheta/'+str(ipix)+'/'+str(jpix)+'/'+str(i)+'/'+str(j)+'/'+d, shape=(self.params['tbins'],), dtype=float )
                        f.create_dataset( '2pt/wtheta/'+str(ipix)+'/'+str(jpix)+'/'+str(i)+'/'+str(j)+'/nntot', shape=(1,), dtype=float )                            
                        f.create_dataset( '2pt/wtheta/'+str(ipix)+'/'+str(jpix)+'/'+str(i)+'/'+str(j)+'/nrtot', shape=(1,), dtype=float )                            
                        f.create_dataset( '2pt/wtheta/'+str(ipix)+'/'+str(jpix)+'/'+str(i)+'/'+str(j)+'/rntot', shape=(1,), dtype=float )                            
                        f.create_dataset( '2pt/wtheta/'+str(ipix)+'/'+str(jpix)+'/'+str(i)+'/'+str(j)+'/rrtot', shape=(1,), dtype=float )                            
        f.close()

        print 'done calcs'

        return calcs

    def run(self):
        """
        This is where all the calculations are done. A mpi pool is set up and the jobs accumulated in setup_jobs() are distributed across nodes.
        """

        if self.comm:
            # Parallel execution
            from .mpi_pool import MPIPool
            # Setup mpi pool
            pool = MPIPool(self.comm)
            # Get job list
            calcs = self.setup_jobs(pool)
            self.comm.Barrier()
            if not pool.is_master():
                # Workers load h5 file (necessary for parallel writing later), wait, then enter queue for jobs
                # f = h5py.File('2pt.h5',mode='r+', driver='mpio', comm=self.comm)
                # self.comm.Barrier()
                # pool.wait()
                print 'slave done',pool.rank
                sys.stdout.flush()
                sys.exit(0)
            # Master opens h5 file (necessary for parallel writing later) and waits for all workers to hit the comm barrier.
            # f = h5py.File('2pt.h5',mode='r+', driver='mpio', comm=self.comm)
            # self.comm.Barrier()
            # Master distributes calculations across nodes.
            pool.map(task, calcs)
            print 'out of main loop',pool.rank
            sys.stdout.flush()
            # # Master waits for everyone to finish, then all close the h5 file and pool is closed.
            # calcs = []
            # for i in range(pool.size):
            #     calcs.append(None)
            # if pool.is_master():
            #     pool.map(task_cleanup, calcs)
            pool.close()
        else:
            # Serial execution
            calcs = self.setup_jobs(None)
            f = h5py.File('2pt.h5',mode='r+')
            map(task, calcs)

    def cleanup(self,tmp):
        """
        Closes h5 to prevent hang.
        """

        f.close()


    def call_treecorr(self,i,j,pix,k):
        """
        This is a dummy function for interaction with the treecorr wrappers.
        """
        #print "Running 2pt analysis on pair {},{},{},{}".format(i, j, pix, k)
        # k==0: xi+-
        # k==1: gammat
        # k==2: wtheta
        
        verbose = 0
        num_threads = self.params['cores_per_task']

        if (k==0): # xi+-
            self.calc_shear_shear(i,j,pix,verbose,num_threads)
        if (k==1): # gammat
            self.calc_pos_shear(i,j,pix,verbose,num_threads)
        if (k==2): # wtheta
            self.calc_pos_pos(i,j,pix,verbose,num_threads)

    def get_zbins_R(self,i,cal):
        """
        Get the lens or source binning, calibration, and weights for a given tomographic bin.
        """ 

        # Open file from nofz stage that contains the catalog tomographic binning indices and read.
        f = h5py.File( self.input_path("nz_source"), mode='r')
        if type(cal)==destest.NoCalib: # Lens catalog
            binning = [f['nofz/lens_zbin'][:]]
        else: # Source catalog
            binning = []
            for zbin_ in ['zbin','zbin_1p','zbin_1m','zbin_2p','zbin_2m']: # Length 5 for unsheared and sheared metacal selections
                binning.append(f['nofz'][zbin_][:])

        # Create tomographic binning mask
        mask = []
        for s in binning:
            mask.append( s == i )

        if type(cal)==destest.NoCalib: # Lens catalog

            # Get random binning as well
            f = h5py.File( self.input_path("nz_source"), mode='r')
            rmask = f['nofz']['ran_zbin'][:] == i
            # Get weights
            w = cal.calibrate('e1', weight_only=True)
            # Return lens binning mask, weights, and random binning mask
            return None, None, mask[self.Dict.ind['u']], w, rmask

        else: # Source catalog

            # Get responses
            R1,c,w = cal.calibrate('e1', mask=mask)
            R2,c,w = cal.calibrate('e2', mask=mask)
            # Return responses, source binning mask and weights            
            return R1, R2, mask[self.Dict.ind['u']], w

    def get_pix_subset(self,ipix,pix,return_neighbor):
        """
        Find the indices of the healpixel subset of the catalog for ipix and optionally all neighboring pixels. pix_ from the catalog is (or should be) pre-sorted for faster searching.
        """

        # Get theta,phi for a healpixel index and return indices of all neighboring pixels (including ipix for auto-correlations).
        theta,phi = hp.pix2ang(self.get_nside(),ipix,nest=True)
        jpix = hp.get_all_neighbours(self.get_nside(),theta,phi,nest=True)
        jpix = np.append(ipix,jpix)

        if return_neighbor:
            # Return objects in pixel ipix and all neighbors 

            pixrange = [] # List of object slices in self and neighboring pixels
            tmp = 0
            for x,jp in enumerate(jpix): # Iterate over neighboring pixels
                tmp2 = np.searchsorted(pix, jp, side='right') - np.searchsorted(pix, jp)
                pixrange.append( np.s_[ int(tmp) : int(tmp + tmp2) ] ) # Individual slices for each neighbor.
                tmp += tmp2

        else:
            # Return objects in pixel ipix 

            pixrange = np.r_[int(np.searchsorted(pix, ipix)) : int(np.searchsorted(pix, ipix, side='right'))] # Find slice (np.r_) corresponding to range of pix_ that contains ipix.

        return pixrange

    def build_catalogs(self,cal,i,ipix,return_neighbor=False):
        """
        Buid catalog subsets in the form of treecorr.Catalog objects for the required tomograhpic and healpixel subsets for this calculation iteration.
        """

        if type(cal)==destest.NoCalib: # lens catalog

            # Get healpix list for lenses
            pix = self.get_lhpix()

            # Get index matching of gold to lens catalog (smaller than gold)
            gmask = cal.selector.get_match()

            # Get tomographic bin masks for lenses and randoms, and weights
            R1,R2,mask,w,rmask = self.get_zbins_R(i,cal)

            # Get ranges for ipix
            pixrange = self.get_pix_subset(ipix,pix[gmask][cal.selector.get_mask()[self.Dict.ind['u']]][mask],return_neighbor)

            # Load ra,dec from gold catalog - source.read is necessary for the raw array to downmatch to lens catalog
            ra  = self.gold_selector.source.read(self.Dict.gold_dict['ra'])[self.Dict.ind['u']][gmask][cal.selector.get_mask()[self.Dict.ind['u']]][mask]
            dec = self.gold_selector.source.read(self.Dict.gold_dict['dec'])[self.Dict.ind['u']][gmask][cal.selector.get_mask()[self.Dict.ind['u']]][mask]
            if not np.isscalar(w):
                w   = w[cal.selector.get_mask()[self.Dict.ind['u']]][mask]

            if np.sum(rmask)>self.params['ran_factor']*np.sum(mask): # Calculate if downsampling is possible
                # Set fixed random seed to make results reproducible
                np.random.seed(seed=self.params['random_seed'])
                # Downsample random catalog to be ran_factor times larger than lenses
                downsample = np.random.choice(np.arange(np.sum(rmask)),self.params['ran_factor']*np.sum(mask),replace=False) # Downsample 

            # Load random ra,dec and calculate healpix values
            ran_ra  = self.ran_selector.get_col(self.Dict.ran_dict['ra'])[self.Dict.ind['u']][rmask][downsample]
            ran_dec = self.ran_selector.get_col(self.Dict.ran_dict['dec'])[self.Dict.ind['u']][rmask][downsample]
            pix     = self.get_hpix(pix=hp.ang2pix(self.params['hpix_nside'],np.pi/2.-np.radians(ran_dec),np.radians(ran_ra),nest=True))

            # Get ranges for ipix
            rpixrange = self.get_pix_subset(ipix,pix,return_neighbor)

            return ra,dec,ran_ra,ran_dec,w,pixrange,rpixrange

        else: # source catalog

            # Get healpix list for sources
            pix = self.get_hpix()

            # Get tomographic bin masks for sources, and responses/weights
            R1,R2,mask,w      = self.get_zbins_R(i,cal)

            # Get ranges for ipix
            pixrange          = self.get_pix_subset(ipix,pix[mask],return_neighbor)

            # Load ra,dec from gold catalog
            ra=self.gold_selector.get_col(self.Dict.gold_dict['ra'])[self.Dict.ind['u']][mask]
            dec=self.gold_selector.get_col(self.Dict.gold_dict['dec'])[self.Dict.ind['u']][mask]

            # Get e1,e2, subtract mean shear, and correct with mean response
            g1=cal.selector.get_col(self.Dict.shape_dict['e1'])[self.Dict.ind['u']][mask]
            g1 = (g1-self.mean_e1[i])/R1
            g2=cal.selector.get_col(self.Dict.shape_dict['e2'])[self.Dict.ind['u']][mask]
            g2 = (g2-self.mean_e2[i])/R2

            return ra,dec,g1,g2,w,pixrange

    def calc_shear_shear(self,i,j,ipix,verbose,num_threads):
        """
        Treecorr wrapper for shear-shear calculations.
        """
        f = h5py.File('2pt_'+str(self.rank)+'.h5',mode='r+')#, driver='mpio', comm=self.comm)

        # Build catalog for tomographic bin i
        ra,dec,g1,g2,w,pixrange = self.build_catalogs(self.source_calibrator,i,ipix)

        # Build treecorr catalog for bin i
        w_ = np.zeros(len(ra))
        w_[pixrange] = w # Set used object's weight
        if np.sum(w_)==0:
            print 'xipm not doing objects for '+str(ipix)+' '+str(i)+' '+str(j)+'. No objects in ipix.'
            sys.stdout.flush()
            for x in range(9):
                f['2pt/xipm/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/tot'][:] = 0.
            f.close()
            return 

        print i,j,ipix,np.sum(w_),pixrange
        print ra[pixrange].min(),ra[pixrange].max(),ra[pixrange].mean()
        print dec[pixrange].min(),dec[pixrange].max(),dec[pixrange].mean()
        print g1[pixrange].min(),g1[pixrange].max(),g1[pixrange].mean()
        print g2[pixrange].min(),g2[pixrange].max(),g2[pixrange].mean()
        # np.save('ra1.npy',ra)
        # np.save('dec1.npy',dec)
        # np.save('g11.npy',g1)
        # np.save('g21.npy',g2)
        # np.save('w_1.npy',w_)

        icat = treecorr.Catalog( g1 = g1, g2   = g2, 
                                 ra = ra, dec  = dec, 
                                 w  = w_, wpos = np.ones(len(ra)), 
                                 ra_units='deg', dec_units='deg')

        # Build catalogs for tomographic bin j
        ra,dec,g1,g2,w,pixrange = self.build_catalogs(self.source_calibrator,j,ipix,return_neighbor=True)

        # Loop over pixels
        for x in range(9):

            # Build treecorr catalog for bin i
            w_ = np.zeros(len(ra))
            w_[pixrange[x]] = w # Set used object's weight
            if np.sum(w_)==0:
                print 'xipm not doing objects for '+str(ipix)+' '+str(x)+' '+str(i)+' '+str(j)+'. No objects in jpix.'
                sys.stdout.flush()
                f['2pt/xipm/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/tot'][:] = 0.
                f.close()
                return 

            print i,j,ipix,x,np.sum(w_),pixrange[x]
            print ra[pixrange[x]].min(),ra[pixrange[x]].max(),ra[pixrange[x]].mean()
            print dec[pixrange[x]].min(),dec[pixrange[x]].max(),dec[pixrange[x]].mean()
            print g1[pixrange[x]].min(),g1[pixrange[x]].max(),g1[pixrange[x]].mean()
            print g2[pixrange[x]].min(),g2[pixrange[x]].max(),g2[pixrange[x]].mean()
            # np.save('ra2.npy',ra)
            # np.save('dec2.npy',dec)
            # np.save('g12.npy',g1)
            # np.save('g22.npy',g2)
            # np.save('w_2.npy',w_)

            jcat = treecorr.Catalog( g1 = g1, g2   = g2,
                                     ra = ra, dec  = dec,
                                     w  = w_,  wpos = np.ones(len(ra)),
                                     ra_units='deg', dec_units='deg')

            print 'xipm doing '+str(np.sum(icat.w))+' '+str(np.sum(jcat.w))+' objects for '+str(ipix)+' '+str(x)+' '+str(i)+' '+str(j)
            sys.stdout.flush()

            # Run calculation
            gg = treecorr.GGCorrelation(nbins=self.params['tbins'], min_sep=self.params['tbounds'][self.Dict.ind['u']], max_sep=self.params['tbounds'][1], sep_units='arcmin', bin_slop=self.params['slop'], verbose=verbose,num_threads=num_threads)
            gg.process_cross(icat,jcat)

            print 'are there pairs',gg.npairs

            # Write output to h5 file
            print 'writing 2pt/xipm/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)
            sys.stdout.flush()
            print f.keys()
            print f['2pt/xipm/'].keys()
            print f['2pt/xipm/'+str(ipix)+'/'].keys()
            print f['2pt/xipm/'+str(ipix)+'/'+str(x)+'/'].keys()
            print f['2pt/xipm/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'].keys()
            print f['2pt/xipm/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+].keys()
            f['2pt/xipm/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/meanlogr'][:] = gg.meanlogr
            f['2pt/xipm/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/xip'][:]      = gg.xip
            f['2pt/xipm/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/xim'][:]      = gg.xim
            f['2pt/xipm/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/npairs'][:]   = gg.npairs
            f['2pt/xipm/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/weight'][:]   = gg.weight
            f['2pt/xipm/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/tot'][:]      = 1
            print 'are there pairs 2',f['2pt/xipm/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/npairs'][:]
            f.close()

        return 

    def calc_pos_shear(self,i,j,ipix,verbose,num_threads):
        """
        Treecorr wrapper for pos-shear calculations.
        """
        f = h5py.File('2pt_'+str(self.rank)+'.h5',mode='r+')#, driver='mpio', comm=self.comm)

        # Build catalog for tomographic bin i
        ra,dec,ran_ra,ran_dec,w,pixrange,rpixrange = self.build_catalogs(self.lens_calibrator,i,ipix)

        # Build treecorr catalog for bin i
        w_ = np.zeros(len(ra))
        w_[pixrange] = w # Set used object's weight
        if np.sum(w_)==0:
            print 'gammat not doing objects for '+str(ipix)+' '+str(i)+' '+str(j)+'. No objects in ipix.'
            sys.stdout.flush()
            for x in range(9):
                f['2pt/gammat/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/ngtot'][:] = 0.
            return

        icat = treecorr.Catalog( ra = ra, dec  = dec, 
                                 w  = w_,  wpos = np.ones(len(ra)), 
                                 ra_units='deg', dec_units='deg')

        w_ = np.zeros(len(ran_ra))
        w_[rpixrange] = 1. # Set used object's weight
        if np.sum(w_)==0:
            print 'gammat not doing objects for '+str(ipix)+' '+str(i)+' '+str(j)+'. No objects in random ipix.'
            sys.stdout.flush()
            for x in range(9):
                f['2pt/gammat/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/ngtot'][:] = 0.
            return

        ircat = treecorr.Catalog( ra = ran_ra, dec  = ran_dec, 
                                  w  = w_,  wpos = np.ones(len(ran_ra)), 
                                  ra_units='deg', dec_units='deg')

        # Build catalogs for tomographic bin j
        ra,dec,g1,g2,w,pixrange = self.build_catalogs(self.source_calibrator,j,ipix,return_neighbor=True)                              

        # Loop over pixels
        for x in range(9):

            # Build treecorr catalog for bin j
            w_ = np.zeros(len(ra))
            w_[pixrange[x]] = w # Set used object's weight
            if np.sum(w_)==0:
                print 'gammat not doing objects for '+str(ipix)+' '+str(x)+' '+str(i)+' '+str(j)+'. No objects in jpix.'
                sys.stdout.flush()
                f['2pt/gammat/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/ngtot'][:] = 0.
                return 

            jcat = treecorr.Catalog( g1 = g1, g2   = g2,
                                     ra = ra, dec  = dec,
                                     w  = w_,  wpos = np.ones(len(ra)),
                                     ra_units='deg', dec_units='deg')

            print 'gammat doing '+str(np.sum(icat.w))+' '+str(np.sum(jcat.w))+' objects for '+str(ipix)+' '+str(x)+' '+str(i)+' '+str(j)
            sys.stdout.flush()

            ng = treecorr.NGCorrelation(nbins=self.params['tbins'], min_sep=self.params['tbounds'][self.Dict.ind['u']], max_sep=self.params['tbounds'][1], sep_units='arcmin', bin_slop=self.params['slop'], verbose=verbose,num_threads=num_threads)
            rg = treecorr.NGCorrelation(nbins=self.params['tbins'], min_sep=self.params['tbounds'][self.Dict.ind['u']], max_sep=self.params['tbounds'][1], sep_units='arcmin', bin_slop=self.params['slop'], verbose=verbose,num_threads=num_threads)
            ng.process_cross(icat,jcat)
            rg.process_cross(ircat,jcat)

            # Write output to h5 file
            print 'writing 2pt/gammat/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)
            sys.stdout.flush()
            f['2pt/gammat/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/meanlogr'][:] = ng.meanlogr
            f['2pt/gammat/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/ngxi'][:]     = ng.xi
            f['2pt/gammat/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/ngxim'][:]    = ng.xi_im
            f['2pt/gammat/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/rgxi'][:]     = rg.xi
            f['2pt/gammat/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/rgxim'][:]    = rg.xi_im
            f['2pt/gammat/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/ngnpairs'][:] = ng.npairs
            f['2pt/gammat/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/ngweight'][:] = ng.weight
            f['2pt/gammat/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/rgnpairs'][:] = rg.npairs
            f['2pt/gammat/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/rgweight'][:] = rg.weight
            f['2pt/gammat/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/ngtot'][:]    = 1
            f['2pt/gammat/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/rgtot'][:]    = 1
            f.close()
        return 

    def calc_pos_pos(self,i,j,ipix,verbose,num_threads):
        """
        Treecorr wrapper for pos-pos calculations.
        """

        f = h5py.File('2pt_'+str(self.rank)+'.h5',mode='r+')#, driver='mpio', comm=self.comm)

        # Build catalog for tomographic bin i
        ra,dec,ran_ra,ran_dec,w,pixrange,rpixrange = self.build_catalogs(self.lens_calibrator,i,ipix)

        # Build treecorr catalog for bin i
        w_ = np.zeros(len(ra))
        w_[pixrange] = w # Set used object's weight
        if np.sum(w_)==0:
            print 'wtheta not doing objects for '+str(ipix)+' '+str(i)+' '+str(j)+'. No objects in ipix.'
            sys.stdout.flush()
            for x in range(9):
                f['2pt/wtheta/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/nntot'][:] = 0.
            return 

        icat = treecorr.Catalog( ra = ra, dec  = dec, 
                                 w  = w_,  wpos = np.ones(len(ra)), 
                                 ra_units='deg', dec_units='deg')

        w_ = np.zeros(len(ran_ra))
        w_[rpixrange] = 1. # Set used object's weight
        if np.sum(w_)==0:
            print 'wtheta not doing objects for '+str(ipix)+' '+str(i)+' '+str(j)+'. No objects in random ipix.'
            sys.stdout.flush()
            for x in range(9):
                f['2pt/wtheta/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/nntot'][:] = 0.
            return 

        ircat = treecorr.Catalog( ra = ran_ra, dec  = ran_dec, 
                                  w  = w_,  wpos = np.ones(len(ran_ra)), 
                                  ra_units='deg', dec_units='deg')

        # Build catalogs for tomographic bin j
        ra,dec,ran_ra,ran_dec,w,pixrange,rpixrange = self.build_catalogs(self.lens_calibrator,j,ipix,return_neighbor=True)  

        # Loop over pixels
        for x in range(9):

            # Build treecorr catalog for bin j
            w_ = np.zeros(len(ra))
            w_[pixrange[x]] = w # Set used object's weight
            if np.sum(w_)==0:
                print 'wtheta not doing objects for '+str(ipix)+' '+str(i)+' '+str(j)+'. No objects in jpix.'
                sys.stdout.flush()
                for x in range(9):
                    f['2pt/wtheta/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/nntot'][:] = 0.
                return 

            jcat = treecorr.Catalog( ra = ra, dec  = dec, 
                                     w  = w_,  wpos = np.ones(len(ra)), 
                                     ra_units='deg', dec_units='deg')

            w_ = np.zeros(len(ran_ra))
            w_[rpixrange[x]] = 1. # Set used object's weight
            if np.sum(w_)==0:
                print 'wtheta not doing objects for '+str(ipix)+' '+str(i)+' '+str(j)+'. No objects in random jpix.'
                sys.stdout.flush()
                for x in range(9):
                    f['2pt/wtheta/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/nntot'][:] = 0.
                return 

            jrcat = treecorr.Catalog( ra = ran_ra, dec  = ran_dec, 
                                      w  = w_,  wpos = np.ones(len(ran_ra)), 
                                      ra_units='deg', dec_units='deg')

            # Run calculation
            nn = treecorr.NNCorrelation(nbins=self.params['tbins'], min_sep=self.params['tbounds'][self.Dict.ind['u']], max_sep=self.params['tbounds'][1], sep_units='arcmin', bin_slop=self.params['slop'], verbose=verbose,num_threads=num_threads)
            rn = treecorr.NNCorrelation(nbins=self.params['tbins'], min_sep=self.params['tbounds'][self.Dict.ind['u']], max_sep=self.params['tbounds'][1], sep_units='arcmin', bin_slop=self.params['slop'], verbose=verbose,num_threads=num_threads)
            nr = treecorr.NNCorrelation(nbins=self.params['tbins'], min_sep=self.params['tbounds'][self.Dict.ind['u']], max_sep=self.params['tbounds'][1], sep_units='arcmin', bin_slop=self.params['slop'], verbose=verbose,num_threads=num_threads)
            rr = treecorr.NNCorrelation(nbins=self.params['tbins'], min_sep=self.params['tbounds'][self.Dict.ind['u']], max_sep=self.params['tbounds'][1], sep_units='arcmin', bin_slop=self.params['slop'], verbose=verbose,num_threads=num_threads)

            nn.process_cross(icat,jcat)
            rn.process_cross(ircat,jcat)
            nr.process_cross(icat,jrcat)
            rr.process_cross(ircat,jrcat)

            # Write output to h5 file
            print 'writing 2pt/wtheta/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)
            sys.stdout.flush()
            f['2pt/wtheta/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/meanlogr'][:] = nn.meanlogr
            f['2pt/wtheta/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/nnnpairs'][:] = nn.npairs
            f['2pt/wtheta/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/nnweight'][:] = nn.weight
            f['2pt/wtheta/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/nrnpairs'][:] = nr.npairs
            f['2pt/wtheta/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/nrweight'][:] = nr.weight
            f['2pt/wtheta/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/rnnpairs'][:] = rn.npairs
            f['2pt/wtheta/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/rnweight'][:] = rn.weight
            f['2pt/wtheta/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/rrnpairs'][:] = rr.npairs
            f['2pt/wtheta/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/rrweight'][:] = rr.weight
            f['2pt/wtheta/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/nntot'][:]    = nn.tot
            f['2pt/wtheta/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/nrtot'][:]    = nr.tot
            f['2pt/wtheta/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/nrtot'][:]    = rn.tot
            f['2pt/wtheta/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/rrtot'][:]    = rr.tot
            f.close()

        return

    def write(self):
        """
        Write data to files. Empty.
        """

        return
