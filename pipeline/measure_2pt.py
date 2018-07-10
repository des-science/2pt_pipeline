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
        if pool is not None:
            if not pool.is_master():
                return calcs

        # Pre-format output h5 file to contain all the necessary paths based on the final calculation list
        if pool is None:
            f = h5py.File('2pt.h5',mode='w')
        else:
            f = h5py.File('2pt.h5',mode='w', driver='mpio', comm=self.comm)
        for i,j,ipix,calc in calcs:
                for jpix in range(9): # There will only ever be 9 pixel pair correlations - the auto-correlation and 8 neighbors
                    if calc==0:
                        for d in ['meanlogr','xip','xim','npairs','weight']:
                            f.create_dataset( '2pt/xipm/'+str(ipix)+'/'+str(jpix)+'/'+str(i)+'/'+str(j)+'/'+d, shape=(self.params['tbins'],), dtype=float )
                    if calc==1:
                        for d in ['meanlogr','ngxi','ngxim','rgxi','rgxim','ngnpairs','ngweight','rgnpairs','rgweight']:
                            f.create_dataset( '2pt/gammat/'+str(ipix)+'/'+str(jpix)+'/'+str(i)+'/'+str(j)+'/'+d, shape=(self.params['tbins'],), dtype=float )
                    if calc==2:
                        for d in ['meanlogr','nnnpairs','nnweight','nrnpairs','nrweight','rnnpairs','rnweight','rrnpairs','rrweight']:
                            f.create_dataset( '2pt/wtheta/'+str(ipix)+'/'+str(jpix)+'/'+str(i)+'/'+str(j)+'/'+d, shape=(self.params['tbins'],), dtype=float )
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
                self.f = h5py.File('2pt.h5',mode='r+', driver='mpio', comm=self.comm)
                self.comm.Barrier()
                pool.wait()
                sys.exit(0)

            # Master opens h5 file (necessary for parallel writing later) and waits for all workers to hit the comm barrier.
            self.f = h5py.File('2pt.h5',mode='r+', driver='mpio', comm=self.comm)
            self.comm.Barrier()
            # Master distributes calculations across nodes.
            pool.map(task, calcs)
            # Master waits for everyone to finish, then all close the h5 file and pool is closed.
            self.f.close()
            pool.close()
        else:
            # Serial execution
            calcs = self.setup_jobs(None)
            self.f = h5py.File('2pt.h5',mode='r+')
            map(task, calcs)

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
            return
            self.calc_shear_shear(i,j,pix,verbose,num_threads)
        if (k==1): # gammat
            self.calc_pos_shear(i,j,pix,verbose,num_threads)
        if (k==2): # wtheta
            self.calc_pos_pos(i,j,pix,verbose,num_threads)

    def get_zbins_R(self,i,cal,shape=True):
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

    def build_catalogs(self,cal,i,ipix,pix,return_neighbor=False):
        """
        Buid catalog subsets in the form of treecorr.Catalog objects for the required tomograhpic and healpixel subsets for this calculation iteration.
        """

        def get_pix_subset(ipix,pix_,return_neighbor):
            """
            Find the indices of the healpixel subset of the catalog for ipix and optionally all neighboring pixels. pix_ from the catalog is (or should be) pre-sorted for faster searching.
            """

            # Get theta,phi for a healpixel index and return indices of all neighboring pixels (including ipix for auto-correlations).
            theta,phi = hp.pix2ang(self.get_nside(),ipix,nest=True)
            jpix = hp.get_all_neighbours(self.get_nside(),theta,phi,nest=True)
            jpix = np.append(ipix,jpix)

            if return_neighbor:
                # Return objects in pixel ipix and all neighbors 

                pixrange = [] # Objects in ipix
                pixrange2 = [] # Lists of objects in neighboring pixels
                tmp = 0
                for x,jp in enumerate(jpix): # Iterate over neighboring pixels
                    pixrange = np.append(pixrange,np.r_[int(np.searchsorted(pix_, jp)) : int(np.searchsorted(pix_, jp, side='right'))]) # Cumulative list of slices (np.r_) of pix_ corresponding to ranges of pixels in jpix list.
                    tmp2 = np.searchsorted(pix_, jp, side='right') - np.searchsorted(pix_, jp)
                    pixrange2.append( np.s_[ int(tmp) : int(tmp + tmp2) ] ) # Individual slices for each neighbor.
                    tmp += tmp2
                pixrange = pixrange.astype(int)

            else:
                # Return objects in pixel ipix 

                pixrange = np.r_[int(np.searchsorted(pix_, ipix)) : int(np.searchsorted(pix_, ipix, side='right'))] # Find slice (np.r_) corresponding to range of pix_ that contains ipix.
                pixrange2 = None

            return pixrange,pixrange2

        if type(cal)==destest.NoCalib: # lens catalog

            # Get index matching of gold to lens catalog (smaller than gold)
            gmask = cal.selector.get_match()
            # Get tomographic bin masks for lenses and randoms, and weights
            R1,R2,mask,w,rmask = self.get_zbins_R(i,cal,shape=False)
            # Get index slices needed for the subset of healpixels in this calculation
            print pix,gmask,cal.selector.get_mask()[self.Dict.ind['u']],mask
            print len(pix),len(gmask),len(cal.selector.get_mask()[self.Dict.ind['u']]),len(mask)
            pixrange,pixrange2 = get_pix_subset(ipix,pix[gmask][cal.selector.get_mask()[self.Dict.ind['u']]][mask],return_neighbor)

            # Load ra,dec from gold catalog - source.read is necessary for the raw array to downmatch to lens catalog
            ra  = self.gold_selector.source.read(self.Dict.gold_dict['ra'])[self.Dict.ind['u']]
            dec = self.gold_selector.source.read(self.Dict.gold_dict['dec'])[self.Dict.ind['u']]

            catlength = len(ra[gmask][cal.selector.get_mask()[self.Dict.ind['u']]][mask][pixrange]) # Length of catalog after masking
            if catlength>0: # Check that objects exist in selection, otherwise return cat = None

                if np.isscalar(w):
                    cat = treecorr.Catalog(ra=ra[gmask][cal.selector.get_mask()[self.Dict.ind['u']]][mask][pixrange],
                                           dec=dec[gmask][cal.selector.get_mask()[self.Dict.ind['u']]][mask][pixrange],
                                           ra_units='deg', dec_units='deg')
                else:
                    cat = treecorr.Catalog(ra=ra[gmask][cal.selector.get_mask()[self.Dict.ind['u']]][mask][pixrange],
                                           dec=dec[gmask][cal.selector.get_mask()[self.Dict.ind['u']]][mask][pixrange],
                                           w = w[pixrange],
                                           ra_units='deg', dec_units='deg')
            else:

                cat = None

            # Load random ra,dec and calculate healpix values
            ra  = self.ran_selector.get_col(self.Dict.ran_dict['ra'])[self.Dict.ind['u']][rmask]
            dec = self.ran_selector.get_col(self.Dict.ran_dict['dec'])[self.Dict.ind['u']][rmask]
            pix = self.get_hpix(pix=hp.ang2pix(self.params['hpix_nside'],np.pi/2.-np.radians(dec),np.radians(ra),nest=True))

            # Get index slices needed for the subset of healpixels in this calculation
            pixrange,rpixrange2 = get_pix_subset(ipix,pix,return_neighbor)

            ranlength = len(ra[pixrange]) # Length of catalog after masking
            if ranlength>0: # Check that objects exist in selection, otherwise return cat = None
                if catlength==0:
                    print 'randoms where no cat',i,ipix
                    rcat=None

                elif ranlength>self.params['ran_factor']*catlength: # Calculate if downsampling is possible
                    # Set fixed random seed to make results reproducible
                    np.random.seed(seed=self.params['random_seed'])
                    # Downsample random catalog to be ran_factor times larger than lenses
                    downsample = np.random.choice(np.arange(ranlength),self.params['ran_factor']*catlength,replace=False) # Downsample 

                    rcat = treecorr.Catalog(ra=ra[pixrange][downsample], 
                                            dec=dec[pixrange][downsample], 
                                            ra_units='deg', dec_units='deg')
                else:
                    rcat = treecorr.Catalog(ra=ra[pixrange], 
                                            dec=dec[pixrange], 
                                            ra_units='deg', dec_units='deg')
            else:

                rcat = None

        else: # source catalog

            # Load ra,dec from gold catalog
            ra=self.gold_selector.get_col(self.Dict.gold_dict['ra'])[self.Dict.ind['u']]
            dec=self.gold_selector.get_col(self.Dict.gold_dict['dec'])[self.Dict.ind['u']]
            # Get tomographic bin masks for sources, and responses/weights
            R1,R2,mask,w = self.get_zbins_R(i,cal)
            # Get index slices needed for the subset of healpixels in this calculation
            pixrange,pixrange2 = get_pix_subset(ipix,pix[mask],return_neighbor)

            # Get e1,e2, subtract mean shear, and correct with mean response
            g1=cal.selector.get_col(self.Dict.shape_dict['e1'])[self.Dict.ind['u']][mask][pixrange]
            g1 = (g1-self.mean_e1[i])/R1
            g2=cal.selector.get_col(self.Dict.shape_dict['e2'])[self.Dict.ind['u']][mask][pixrange]
            g2 = (g2-self.mean_e2[i])/R2

            if len(g1)>0: # Check there are objects in this selection, otherwise return cat = None
                if np.isscalar(w):
                    cat = treecorr.Catalog(g1=g1, g2=g2, ra=ra[mask][pixrange], 
                                            dec=dec[mask][pixrange], ra_units='deg', dec_units='deg')
                else:
                    cat = treecorr.Catalog(g1=g1, g2=g2, w=w, ra=ra[mask][pixrange], 
                                            dec=dec[mask][pixrange], ra_units='deg', dec_units='deg')
            else:
                cat = None
                return cat,pixrange2

        if type(cal)==destest.NoCalib:
            return cat,rcat,pixrange2,rpixrange2
        else:
            return cat,pixrange2

    def calc_shear_shear(self,i,j,ipix,verbose,num_threads):
        """
        Treecorr wrapper for shear-shear calculations.
        """

        # Get healpix list for sources
        pix = self.get_hpix()
        # Build catalogs for tomographic bin i
        icat,pixrange = self.build_catalogs(self.source_calibrator,i,ipix,pix)
        # Build catalogs for tomographic bin j
        jcat,pixrange = self.build_catalogs(self.source_calibrator,j,ipix,pix,return_neighbor=True)

        if (icat is None) or (jcat is None): # No objects in selection
            print 'xipm not doing objects for '+str(ipix)+' '+str(i)+' '+str(j)+'. ',icat is None,jcat is None
            return 

        # Loop over pixels
        for x in range(9):
            jcat.wpos[:] = 0. # Set up dummy weight to preserve tree
            jcat.wpos[pixrange[x]] = 1. # Set used objects dummy weight to 1
            print 'xipm doing '+str(len(icat.ra))+' '+str(np.sum(jcat.wpos))+' objects for '+str(ipix)+' '+str(x)+' '+str(i)+' '+str(j)
            # Run calculation
            if np.sum(jcat.wpos)==0:
                continue
            gg = treecorr.GGCorrelation(nbins=self.params['tbins'], min_sep=self.params['tbounds'][self.Dict.ind['u']], max_sep=self.params['tbounds'][1], sep_units='arcmin', bin_slop=self.params['slop'], verbose=verbose,num_threads=num_threads)
            gg.process_cross(icat,jcat)

            # Write output to h5 file
            print 'writing 2pt/xipm/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)
            self.f['2pt/xipm/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/meanlogr'][:] = gg.meanlogr
            self.f['2pt/xipm/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/xip'][:]      = gg.xip
            self.f['2pt/xipm/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/xim'][:]      = gg.xim
            self.f['2pt/xipm/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/npairs'][:]   = gg.npairs
            self.f['2pt/xipm/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/weight'][:]   = gg.weight

        return 

    def calc_pos_shear(self,i,j,ipix,verbose,num_threads):
        """
        Treecorr wrapper for pos-shear calculations.
        """

        # Get healpix list for lenses
        pix = self.get_lhpix()
        # Build catalogs for tomographic bin i
        icat,ircat,pixrange,rpixrange = self.build_catalogs(self.lens_calibrator,i,ipix,pix)       
        # Get healpix list for sources
        pix = self.get_hpix()
        # Build catalogs for tomographic bin j
        jcat,pixrange = self.build_catalogs(self.source_calibrator,j,ipix,pix,return_neighbor=True)                                  
        if (icat is None) or (jcat is None) or (ircat is None): # No objects in selection
            print 'gammat not doing objects for '+str(ipix)+' '+str(i)+' '+str(j)+'. ',icat is None,jcat is None,ircat is None
            return 

        # print icat,jcat,ircat

        # Loop over pixels
        for x in range(9):
            jcat.wpos[:] = 0. # Set up dummy weight to preserve tree
            jcat.wpos[pixrange[x]] = 1. # Set used objects dummy weight to 1
            print 'gammat doing '+str(len(icat.ra))+' '+str(np.sum(jcat.wpos))+' objects for '+str(ipix)+' '+str(x)+' '+str(i)+' '+str(j)
            # Run calculation
            if np.sum(jcat.wpos)==0:
                print pixrange[x]
                continue
            ng = treecorr.NGCorrelation(nbins=self.params['tbins'], min_sep=self.params['tbounds'][self.Dict.ind['u']], max_sep=self.params['tbounds'][1], sep_units='arcmin', bin_slop=self.params['slop'], verbose=verbose,num_threads=num_threads)
            rg = treecorr.NGCorrelation(nbins=self.params['tbins'], min_sep=self.params['tbounds'][self.Dict.ind['u']], max_sep=self.params['tbounds'][1], sep_units='arcmin', bin_slop=self.params['slop'], verbose=verbose,num_threads=num_threads)
            ng.process_cross(icat,jcat)
            rg.process_cross(ircat,jcat)

            # Write output to h5 file
            print 'writing 2pt/gammat/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)
            self.f['2pt/gammat/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/meanlogr'][:] = ng.meanlogr
            self.f['2pt/gammat/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/ngxi'][:]     = ng.xi
            self.f['2pt/gammat/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/ngxim'][:]    = ng.xi_im
            self.f['2pt/gammat/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/rgxi'][:]     = rg.xi
            self.f['2pt/gammat/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/rgxim'][:]    = rg.xi_im
            self.f['2pt/gammat/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/ngnpairs'][:] = ng.npairs
            self.f['2pt/gammat/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/ngweight'][:] = ng.weight
            self.f['2pt/gammat/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/rgnpairs'][:] = rg.npairs
            self.f['2pt/gammat/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/rgweight'][:] = rg.weight

        return 

    def calc_pos_pos(self,i,j,ipix,verbose,num_threads):
        """
        Treecorr wrapper for pos-pos calculations.
        """

        # Get healpix list for lenses
        pix = self.get_lhpix()
        # Build catalogs for tomographic bin i
        icat,ircat,pixrange,rpixrange = self.build_catalogs(self.lens_calibrator,i,ipix,pix)
        # Build catalogs for tomographic bin i
        jcat,jrcat,pixrange,rpixrange = self.build_catalogs(self.lens_calibrator,i,ipix,pix,return_neighbor=True)

        if (icat is None) or (jcat is None) or (ircat is None) or (jrcat is None): # No objects in selection
            print 'wtheta not doing objects for '+str(ipix)+' '+str(i)+' '+str(j)+'. ',icat is None,jcat is None,ircat is None,jrcat is None
            return 

        # Loop over pixels
        for x in range(9):
            jcat.wpos[:] = 0. # Set up dummy weight to preserve tree
            jcat.wpos[pixrange[x]] = 1. # Set used objects dummy weight to 1
            print 'wtheta doing '+str(len(icat.ra))+' '+str(np.sum(jcat.wpos))+' objects for '+str(ipix)+' '+str(x)+' '+str(i)+' '+str(j)
            if np.sum(jcat.wpos)==0:
                continue

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
            self.f['2pt/wtheta/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/meanlogr'][:] = nn.meanlogr
            self.f['2pt/wtheta/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/nnnpairs'][:] = nn.npairs
            self.f['2pt/wtheta/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/nnweight'][:] = nn.weight
            self.f['2pt/wtheta/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/nrnpairs'][:] = nr.npairs
            self.f['2pt/wtheta/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/nrweight'][:] = nr.weight
            self.f['2pt/wtheta/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/rnnpairs'][:] = rn.npairs
            self.f['2pt/wtheta/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/rnweight'][:] = rn.weight
            self.f['2pt/wtheta/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/rrnpairs'][:] = rr.npairs
            self.f['2pt/wtheta/'+str(ipix)+'/'+str(x)+'/'+str(i)+'/'+str(j)+'/rrweight'][:] = rr.weight

        return

    def write(self):
        """
        Write data to files. Empty.
        """

        return
