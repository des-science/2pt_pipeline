import numpy as np
import twopoint
import fitsio as fio
import healpy as hp
from numpy.lib.recfunctions import append_fields, rename_fields
from .stage import PipelineStage, NOFZ_NAMES
import subprocess
import os
import warnings
import destest
import yaml
import pdb
import importlib
import h5py

class ParamError(Exception):
  def __init__(self, value):
    self.value = value
  def __str__(self):
    return repr(self.value)

destest_dict_ = {
    'output_exists' : True,
    'use_mpi'       : False,
    'source'        : 'hdf5',
    'dg'            : 0.01
    }

def create_destest_yaml( filename, cal_type, group, table, select_path, name_dict ):
    """
    Creates the input dictionary structure from a passed dictionary rather than reading froma yaml file.
    """

    params = yaml.load(open(filename))

    destest_dict = destest_dict_
    destest_dict['load_cache'] = params['load_cache']
    destest_dict['output'] = params['output']
    destest_dict['filename'] = params['datafile']
    destest_dict['param_file'] = filename
    destest_dict['cal_type'] = cal_type
    destest_dict['group'] = group
    destest_dict['table'] = table
    destest_dict['select_path'] = select_path
    destest_dict['e'] = [name_dict.shape_dict['e1'],name_dict.shape_dict['e2']]
    destest_dict['Rg'] = [name_dict.shape_dict['m1'],name_dict.shape_dict['m2']]

    return destest_dict

def load_catalog(pipe_params, cal_type, group, table, select_path, name_dict, inherit=None, return_calibrator=None):
    """
    Loads data access and calibration classes from destest for a given yaml setup file.
    """
    # Input yaml file defining catalog
    params = create_destest_yaml(pipe_params, cal_type, group, table, select_path, name_dict)
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

class nofz(PipelineStage):
    name = "nofz"
    inputs = {}
    outputs = {
        "weight"        : "weight.npy"          ,
        "nz_source"     : "nz_source_zbin.h5"   ,
        "nz_lens"       : "nz_lens_zbin.npy"    ,
        "nz_source_txt" : "source.nz"           ,
        "nz_lens_txt"   : "lens.nz"             ,
        "gold_idx"      : "gold_idx.npy"        ,
        "lens_idx"      : "lens_idx.npy"        ,
        "ran_idx"       : "ran_idx.npy"         ,
        "randoms"       : "randoms_zbin.npy"    ,
        "cov_ini"       : "cov.ini"             ,
        "2pt"           : "2pt.fits"            ,
        "metadata"      : "metadata.yaml"
    }

    def __init__(self, param_file):
        """
        Produces tomographic binning, n(z)s, and metadata from input catalogs.
        """
        super(nofz,self).__init__(param_file)
        
        # A dictionary to homogenize names of columns in the hdf5 master catalog 
        self.Dict = importlib.import_module('.'+self.params['dict_file'],'pipeline')
        print 'using dictionary: ',self.params['dict_file']
                
        # Load data and calibration classes
        self.source_selector, self.source_calibrator = load_catalog(self.params, 'mcal', params['source_group'], params['source_table'], params['source_path'], self.Dict, return_calibrator=destest.MetaCalib)
        self.lens_selector, self.lens_calibrator     = load_catalog(self.params, None, params['lens_group'], params['lens_table'], params['lens_path'], self.Dict, return_calibrator=destest.NoCalib)
        self.gold_selector = load_catalog(self.params, 'mcal', params['gold_group'], params['gold_table'], params['gold_path'], self.Dict, inherit=self.source_selector)
        self.pz_selector   = load_catalog(self.params, 'mcal', params['pz_group'], params['pz_table'], params['pz_path'], self.Dict,   inherit=self.source_selector)
        self.ran_selector  = load_catalog(self.params, None, params['ran_group'], params['ran_table'], params['ran_path'], self.Dict)

        self.Dict.ind = self.Dict.index_dict #a dictionary that takes unsheared,sheared_1p/1m/2p/2m as u-1-2-3-4 to deal with tuples of values returned by get_col()

        # Setup n(z) array binning for sources
        if self.params['pdf_type']!='pdf':
            # Define z binning of returned n(z)s

            self.z       = np.linspace(0., self.params['nzmax'], self.params['nzbins']+1)
            self.z       = (self.z[1:] + self.z[:-1]) / 2. + 1e-4 # 1e-4 for buzzard redshift files
            self.dz      = self.z[1] - self.z[0] # N(z) binning width
            self.binlow  = self.z - self.dz/2. # Lower bin edges
            self.binhigh = self.z + self.dz/2. # Upper bin edges

        else:
            # Adopt pdf binning

            raise ParamError('Not updated to work with full pdfs.')

        # Setup tomographic bin edges for sources
        if hasattr(self.params['zbins'], "__len__"):
            # Provided array of bin edges in yaml
            self.tomobins = len(self.params['zbins']) - 1 # Number of tomographic bins
            self.binedges = self.params['zbins'] # Bin edges
        else:
            # Provided number of bins in yaml
            self.tomobins = self.params['zbins']
            self.binedges = self.find_bin_edges(self.pz_selector.get_col(self.Dict.pz_dict['pzbin']), self.tomobins, w = self.shape['weight'][self.mask])
            
        # Setup tomographic bin edges for lenses
        if self.params['lens_yaml'] != 'None':
            if hasattr(self.params['lens_zbins'], "__len__"):
                # Provided array of bin edges in yaml
                self.lens_tomobins = len(self.params['lens_zbins']) - 1 
                self.lens_binedges = self.params['lens_zbins']
            else:
                # Provided number of bins in yaml
                self.lens_tomobins = self.params['lens_zbins']
                self.lens_binedges = self.find_bin_edges(self.lens_pz['pzbin'], self.lens_tomobins, w = self.lens['weight']) 
        
        return

    def run(self):
        """
        Run the nofz module to produce n(z)s and metadata.
        """

        # Get the PZ binning and stacking arrays
        pzbin   = self.pz_selector.get_col(self.Dict.pz_dict['pzbin'])
        print 'len pzbin',len(pzbin),len(pzbin[0])
        pzstack = self.pz_selector.get_col(self.Dict.pz_dict['pzstack'])[self.Dict.ind['u']]

        if self.params['pdf_type']!='pdf': 
            # Get binning and n(z) by stacking a scalar derived from pdf

            print pzbin, pzstack,self.binedges,self.tomobins

            zbin, self.nofz = self.build_nofz_bins(
                self.tomobins, # Number of tomographic bins
                self.binedges, # Tomographic bin edges
                pzbin, # Array by which to bin
                pzstack, # Array by which to stack
                self.params['pdf_type'], # Type of stacking
                None, # Weight array: shape weight * response
                shape=True) # Is this a source operation?

        else: 

            raise ParamError('Not updated to work with full pdfs.')

        # Calculate sigma_e and n_eff
        self.get_sige_neff(zbin,self.tomobins)

        # Write source tomographic binning indicies to file for use later in the pipeline
        f = h5py.File( self.output_path("nz_source"), mode='w')
        for zbin_,zname in tuple(zip(zbin,['zbin','zbin_1p','zbin_1m','zbin_2p','zbin_2m'])):
            f.create_dataset( 'nofz/'+zname, maxshape=(2*len(zbin_),), shape=(len(zbin_),), dtype=zbin_.dtype, chunks=(len(zbin_)/10,) )
            f['nofz/'+zname][:] = zbin_

        # Get the lens PZ binning and stacking arrays and weights
        pzbin   = self.lens_selector.get_col(self.Dict.lens_pz_dict['pzbin'])
        pzstack = self.lens_selector.get_col(self.Dict.lens_pz_dict['pzstack'])[self.Dict.ind['u']]
        weight  = self.lens_calibrator.calibrate(self.Dict.lens_pz_dict['weight'],weight_only=True) 
                
        if self.params['lens_yaml'] != 'None':
            # Calculate lens n(z)s and write to file
            lens_zbin, self.lens_nofz = self.build_nofz_bins(
                                         self.lens_tomobins, 
                                         self.lens_binedges,
                                         pzbin,
                                         pzstack,
                                         self.params['lens_pdf_type'],
                                         weight)
            lens_zbin = lens_zbin[0]

            # Write lens tomographic binning indicies to file for use later in the pipeline
            f = h5py.File( self.output_path("nz_source"), mode='r+')
            f.create_dataset( 'nofz/lens_zbin', maxshape=(len(lens_zbin),), shape=(len(lens_zbin),), dtype=lens_zbin.dtype, chunks=(len(lens_zbin)/10,) )
            f['nofz/lens_zbin'][:] = lens_zbin

            # Calculate and write random tomographic binning indicies to file for use later in the pipeline
            pzbin = self.ran_selector.get_col(self.Dict.ran_dict['ranbincol'])[self.Dict.ind['u']]
            ran_binning = np.digitize(pzbin, self.lens_binedges, right=True) - 1

            f.create_dataset( 'nofz/ran_zbin', maxshape=(len(ran_binning),), shape=(len(ran_binning),), dtype=ran_binning.dtype, chunks=(len(ran_binning)/10,) )
            f['nofz/ran_zbin'][:] = ran_binning

            # Calculate lens n_eff
            self.get_lens_neff( lens_zbin, self.lens_tomobins, weight)

        f.close()

    def write(self):
        """
        Write lens and source n(z)s to twopoint fits file.
        """

        # Create source twopoint number density object
        nz_source = twopoint.NumberDensity(
                     NOFZ_NAMES[0],
                     self.binlow, 
                     self.z, 
                     self.binhigh, 
                     [self.nofz[i,:] for i in range(self.tomobins)])

        # Add metatdata
        nz_source.ngal      = self.neff
        nz_source.sigma_e   = self.sigma_e
        nz_source.area      = self.area
        kernels             = [nz_source]
        np.savetxt(self.output_path("nz_source_txt"), np.vstack((self.binlow, self.nofz)).T)

        # Doing calculations on lenses, so include them
        if self.params['lens_yaml'] != 'None':
            # Create lens twopoint number density object
            nz_lens      = twopoint.NumberDensity(
                            NOFZ_NAMES[1], 
                            self.binlow, 
                            self.z, 
                            self.binhigh, 
                            [self.lens_nofz[i,:] for i in range(self.lens_tomobins)])

            # Add metatdata
            nz_lens.ngal = self.lens_neff
            nz_lens.area = self.area
            kernels.append(nz_lens)
            np.savetxt(self.output_path("nz_lens_txt"), np.vstack((self.binlow, self.lens_nofz)).T)

        # Write to twopoint fits file
        data             = twopoint.TwoPointFile([], kernels, None, None)
        data.to_fits(self.output_path("2pt"), clobber=True)

        # Write metadata to yaml file for use in further pipeline stages
        self.write_metadata()

    def write_metadata(self):
        """
        Write metadata to yaml file for use in further pipeline stages.
        """

        # Define dictionary structure
        data = {
            "neff": self.neff,
            "neffc": self.neffc,
            "tomobins": self.tomobins,
            "sigma_e": self.sigma_e,
            "sigma_ec": self.sigma_ec,
            "mean_e1":self.mean_e1,
            "mean_e2":self.mean_e2,
            "area": self.area,
            "repository_version:": find_git_hash(),
        }

        # Add source bin edges
        if type(self.binedges) is list:
            data.update({ "source_bins" : self.binedges })
        else:
            data.update({ "source_bins" : self.binedges.tolist() })

        # Add lens bin information
        if self.params['lens_yaml'] != 'None':
            data.update({ "lens_neff" : self.lens_neff,
                          "lens_tomobins" : self.lens_tomobins,
                          "lens_bins" : self.lens_binedges })
        print data
        # Save dict to yaml file.
        filename = self.output_path('metadata')
        open(filename, 'w').write(yaml.dump(data))

    def build_nofz_bins(self, zbins, edge, bin_col, stack_col, pdf_type, weight, shape=False):
        """
        Build the n(z)s.
        """

        # Create tomographic bin indicies from bin edges.
        if (~shape)|(self.params['has_sheared']):

            # Loop over unsheared and sheared catalogs (bin_col list)
            xbins0=[]
            for x in bin_col:
                # Digitize the bin indices from the edges
                xbins0.append(np.digitize(x, edge, right=True) - 1)
            xbins = xbins0[0]

        else:

            raise ParamError('Not updated to support non-metacal catalogs.')

        # Create empty n(z) array
        nofz  = np.zeros((zbins, len(self.z)))

        # N(z) is created from stacking scalar value derived from the pdf
        if (pdf_type == 'sample') | (pdf_type == 'rm'):

            # Photo-z stacking for redmagic (or any catalog that reports a gaussian photo-z mean + width)
            if pdf_type == 'rm':
                # Set fixed random seed to make results reproducible
                np.random.seed(seed=self.params['random_seed'])
                # Stack value derived as random draw from gaussian reported photo-z
                stack_col = np.random.normal(stack_col, self.lens_selector.get_col(self.Dict.lens_pz_dict['pzerr'])[self.Dict.ind['u']])

            # Stack scalar values into n(z) looping over tomographic bins
            for i in range(zbins):
                # Get array masks for the tomographic bin for unsheared and sheared catalogs
                print i,xbins,edge,bin_col,len(xbins),np.sum(xbins == i),np.sum((bin_col[0]>edge[i])&(bin_col[0]<edge[i+1]))
                mask        =  (xbins == i)
                if shape:
                    if self.params['has_sheared']:
                        mask_1p = (xbins0[1] == i)
                        mask_1m = (xbins0[2] == i)
                        mask_2p = (xbins0[3] == i)
                        mask_2m = (xbins0[4] == i)

                        weight_ = self.source_calibrator.calibrate(self.Dict.shape_dict['e1'],mask=[mask],return_wRg=True) # This returns an array of (Rg1+Rg2)/2*w for weighting the n(z) 
                        print 'weight',weight_
                        
                    else:

                        raise ParamError('Not updated to support non-metacal catalogs.')

                else:

                    weight_ = weight 

                # Stack n(z)
                if np.isscalar(weight_):

                    nofz[i,:],b =  np.histogram(stack_col[mask], bins=np.append(self.binlow, self.binhigh[-1]))
                else:

                    nofz[i,:],b =  np.histogram(stack_col[mask], bins=np.append(self.binlow, self.binhigh[-1]), weights=weight_)

                nofz[i,:]   /= np.sum(nofz[i,:]) * self.dz

        # Stacking pdfs
        elif pdf_type == 'pdf':

             raise ParamError('Not updated to work with full pdfs.')

        return xbins0, nofz

    def find_bin_edges(self, x, nbins, w=None):
        """
        For an array x, returns the boundaries of nbins equal (possibly weighted by w) bins.
        From github.com/matroxel/destest.

        This is probably really slow for DES Y3 size catalogs. There's a better way built into destest, but probably won't ever use this, so haven't ported over.
        """

        if w is None:
            xs = np.sort(x)
            r  = np.linspace(0., 1., nbins + 1.) * (len(x) - 1)
            return xs[r.astype(int)]

        fail = False
        ww   = np.sum(w) / nbins
        i    = np.argsort(x)
        k    = np.linspace(0.,1., nbins + 1.) * (len(x) - 1)
        k    = k.astype(int)
        r    = np.zeros((nbins + 1))
        ist  = 0
        for j in xrange(1,nbins):
            if k[j]  < r[j-1]:
                print 'Random weight approx. failed - attempting brute force approach'
                fail = True
                break

            w0 = np.sum(w[i[ist:k[j]]])
            if w0 <= ww:
                for l in xrange(k[j], len(x)):
                    w0 += w[i[l]]
                    if w0 > ww:
                        r[j] = x[i[l]]
                        ist  = l
                        break
            else:
                for l in xrange(k[j], 0, -1):
                    w0 -= w[i[l]]
                    if w0 < ww:
                        r[j] = x[i[l]]
                        ist  = l
                        break

        if fail:
            ist = np.zeros((nbins+1))
            ist[0]=0
            for j in xrange(1, nbins):
                wsum = 0.
                for k in xrange(ist[j-1].astype(int), len(x)):
                    wsum += w[i[k]]
                    if wsum > ww:
                        r[j]   = x[i[k-1]]
                        ist[j] = k
                        break

        r[0]  = x[i[0]]
        r[-1] = x[i[-1]]

        return r

    def get_lens_neff(self, zbin, tomobins, weight):
        """
        Calculate neff for lens catalog.
        """

        # Check if area is defined - else retrieve it
        if not hasattr(self,'area'):
            self.get_area()

        # Define empty arrays that will be saved
        self.lens_neff = []

        for i in range(tomobins):
            # Loop over source tomographic bins

            # get object selection for tomographic bin i
            mask = (zbin == i)

            if np.isscalar(weight):
                # If weight is a scalar (no weights), calculate simple n_raw

                self.lens_neff.append( np.asscalar( np.sum( mask ) / ( self.area * 60.**2 ) ) )

            else:

                # Calculate components of the sigma_e and n_eff calculations
                a    = np.sum( weight[mask]    )**2
                b    = np.sum( weight[mask]**2 )
                # print np.sum(weight[mask]),'objects found in this bin',weight[mask]
                
                self.lens_neff.append( np.asscalar( a / b / ( self.area * 60.**2 ) ) )

    def get_sige_neff(self, zbin, tomobins):
        """
        Calculate sigma_e and n_eff for for the source tomographic bins.
        """

        # Check if area is defined - else retrieve it
        if not hasattr(self,'area'):
            self.get_area()

        # Define empty arrays that will be saved
        self.mean_e1  = []
        self.mean_e2  = []
        self.sigma_e  = [] # Heymans versions - these are typically used in covariance calculations
        self.neff     = []
        self.sigma_ec = [] # Chang versions - better at a theoretical level
        self.neffc    = []

        for i in range(tomobins):
            # Loop over source tomographic bins
            print 'Doing sige and neff for source zbin',i

            if self.params['has_sheared']:
                # Select objects in tomographic bin i
                mask    = (zbin[0] == i)
                mask_1p = (zbin[1] == i)
                mask_1m = (zbin[2] == i)
                mask_2p = (zbin[3] == i)
                mask_2m = (zbin[4] == i)

                # Calculate mean reponse in tomographic bin i and get weight vector
                R,c,w = self.source_calibrator.calibrate(self.Dict.shape_dict['e1'],mask=[mask,mask_1p,mask_1m,mask_2p,mask_2m])
                if type(w) is list:
                    w = w[0]

                # print np.sum(mask),'objects found in this bin'
                # Select objects in bin and get e and e cov arrays
                e1  = self.source_selector.get_col(self.Dict.shape_dict['e1'], 
                                                  nosheared=True)[self.Dict.ind['u']][mask]
                e2  = self.source_selector.get_col(self.Dict.shape_dict['e2'], 
                                                  nosheared=True)[self.Dict.ind['u']][mask]
                s   = R
                var = self.source_selector.get_col(self.Dict.shape_dict['cov00'], nosheared=True)[self.Dict.ind['u']][mask] + self.source_selector.get_col(self.Dict.shape_dict['cov11'], nosheared=True)[self.Dict.ind['u']][mask]
                # Regularize variance for small number of ill-defined covariances
                var[var>2] = 2.
            
            else:

                raise ParamError('Not updated to support non-metacal catalogs.')

            if np.isscalar(w):
                # Calculate mean shear without calibration factor
                self.mean_e1.append( np.asscalar( np.average(e1) ) )
                self.mean_e2.append( np.asscalar( np.average(e2) ) )
                # Calculate components of the sigma_e and n_eff calculations
                sum_we2_1 = np.sum( w**2 * ( e1 - self.mean_e1[i] )**2 )
                sum_we2_2 = np.sum( w**2 * ( e2 - self.mean_e2[i] )**2 )
                sum_w2    = np.sum( mask )
                sum_ws    = np.sum( mask ) * s
                sum_w     = np.sum( mask )
                sum_w2s2  = np.sum( mask ) * s**2
            else:
                # Calculate mean shear without calibration factor
                self.mean_e1.append( np.asscalar( np.average(e1, weights=w) ) )
                self.mean_e2.append( np.asscalar( np.average(e2, weights=w) ) )
                # Calculate components of the sigma_e and n_eff calculations
                sum_we2_1 = np.sum( w**2 * ( e1 - self.mean_e1[i] )**2 )
                sum_we2_2 = np.sum( w**2 * ( e2 - self.mean_e2[i] )**2 )
                sum_w2    = np.sum( w**2  )
                sum_ws    = np.sum( w * s )
                sum_w     = np.sum( w     )
                sum_w2s2  = np.sum( w**2 * s**2 )
            
            print 'neffsige',i,np.sum(mask),np.sum(mask_1p),np.mean,sum_w,sum_w2

            # Calculate sigma_e 
            self.sigma_e.append( np.sqrt( (sum_we2_1 / sum_ws**2 + sum_we2_2 / sum_ws**2) 
                                          * (sum_w**2 / sum_w2) / 2. ) )
            self.sigma_ec.append( np.sqrt( np.sum( w**2 * (e1**2 + e2**2 - var) ) 
                                           / ( 2. * sum_w2s2 ) 
                                          ) 
                                 )

            # Calculate n_eff
            self.neff.append( sum_w**2 / sum_w2 / ( self.area * 60. * 60. ) )
            print '.......',w,np.sum(mask),sum_w**2,sum_w2,self.area * 60. * 60.,self.area
            self.neffc.append( ( self.sigma_ec[i]**2 * sum_ws**2 ) 
                                 / np.sum( w**2 * ( s**2 * self.sigma_ec[i]**2 + var / 2. ) )
                               / self.area / 60**2 
                              )

    def get_area(self):
        """
        Retrieve area from yaml file or calculate it if not provided.
        """

        if hasattr(self,'area'):
            # Area already calculated or read in elsewhere.
            return

        if 'area' not in self.params:
            # Area not provided in yaml - calculating with healpix.

            print 'Calculating area via healpixel counting -- very inaccurate, you should be worried and provide a better effective area estimate.'

            # Calculate pixel positions
            ra   = self.gold_selector.get_col(self.Dict.gold_dict['ra'])[self.Dict.ind['u']]
            dec  = self.gold_selector.get_col(self.Dict.gold_dict['dec'])[self.Dict.ind['u']]
            pix  = hp.ang2pix(4096, np.pi/2. - np.radians(dec),
                                    np.radians(ra), nest=True)
            # Calculate area of pixel in deg^2
            area = hp.nside2pixarea(4096) * ( 180. / np.pi )**2
            # Get number of used pixels
            mask = np.bincount(pix) > 0
            # Multiple to get final area
            self.area=float( np.sum(mask) * area )
        
        else:
            # Area provided in yaml

            self.area = self.params['area']

def find_git_hash():
    try:
        dirname = os.path.dirname(os.path.abspath(__file__))
        head = subprocess.check_output("cd {0}; git show-ref HEADS".format(dirname), shell=True)
    except subprocess.CalledProcessError:
        head = "UNKNOWN"
        warnings.warn("Unable to find git repository commit ID in {}".format(dirname))
    return head.split()[0]


