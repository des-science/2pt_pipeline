import numpy as np
import twopoint
import healpy
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
        Produces n(z)s from input catalogs.
        """
        super(nofz,self).__init__(param_file)
        
        
        #using a dictionary that changes names of columns in the hdf5 master catalog to simpler 
        self.Dict = importlib.import_module('.'+self.params['dict_file'],'pipeline')
        print 'using dictionary: ',self.params['dict_file']
                
        print 'mcal selector'
        #mcal_file = '/global/homes/s/seccolf/des-science/2pt_pipeline/destest_mcal.yaml'
        mcal_file = 'destest_mcal.yaml'
        params_mcal = yaml.load(open(mcal_file))
        params_mcal['param_file'] = mcal_file
        source_mcal = destest.H5Source(params_mcal)
        self.selector_mcal = destest.Selector(params_mcal,source_mcal)
        self.calibrator = destest.MetaCalib(params_mcal,self.selector_mcal)
        #now, using selector_mcal.get_col(col) should return a column from the catalog for column name col with the cuts specified by the destest_mcal.yaml file

        print 'lens selector'
        lens_file = 'destest_redmagic.yaml'
        params_lens = yaml.load(open(lens_file))
        params_lens['param_file'] = lens_file
        source_lens = destest.H5Source(params_lens)
        self.selector_lens = destest.Selector(params_lens,source_lens)
        self.calibrator_lens = destest.NoCalib(params_lens,self.selector_lens)

        print 'random selector'
        random_file = 'destest_random.yaml'
        params_random = yaml.load(open(random_file))
        params_random['param_file'] = random_file
        source_random = destest.H5Source(params_random)
        self.selector_random = destest.Selector(params_random,source_random)
        
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
        self.selector_pz = destest.Selector(params_pz,source_pz,inherit=self.selector_mcal)
        #now you get columns by doing col=selector_pz.get_col(Dict.pz_dict['pzbin']) for instance
        
        self.Dict.ind = self.Dict.index_dict #a dictionary that takes unsheared,sheared_1p/1m/2p/2m as u-1-2-3-4 to deal with tuples of values returned by get_col()


        # snr = self.selector_mcal.get_col('snr')
        # for i in range(5):
        #     print np.max(snr[i]),np.min(snr[i])

        # Load data #comment it now as a test
        #self.load_data() #Lucas: maybe will have to get rid of this entirely

        #once we actually start using weights, use the few lines below
        S = len(self.selector_pz.get_col(self.Dict.pz_dict['pzbin'],nosheared=True))
        #print S
        self.weight = np.ones(S)
        #print '\nweights done\n'
        ''' 
        if 'pzbin_col' in self.gold.dtype.names:
            print 'ignoring any specified bins, since a bin column has been supplied in gold file'

        self.weight = np.sqrt(self.selector_pz.get_col('weight') * selector_mcal.get_col('weight'))
        # Construct new weight and cache - move to catalog
        if 'weight' in self.pz.dtype.names:
            print 'I will try to access the HDF5 cat using selector.get_col() right now\n\n'
            self.weight = np.sqrt(selector_pz.get_col('weight') * selector_mcal.get_col('weight')) #my modification
            #self.weight = np.sqrt(self.pz['weight'] * self.shape['weight']) #Lucas: every access to self.pz or shape columns should be replaced by an access of the hdf5 cat itself. Use functions from destest. So this should be switched by selector.get_col(column_name)
        else:
            self.weight = self.shape['weight']
        
        filename = self.output_path("weight")
        np.save(filename, np.vstack((self.gold['objid'], self.weight)).T)
        # deal with photo-z weights for lenses later...
        '''

        
        # Setup binning
        if self.params['pdf_type']!='pdf': 
            self.z       = (np.linspace(0.,4.,401)[1:]+np.linspace(0.,4.,401)[:-1])/2.+1e-4 # 1e-4 for buzzard redshift files
            self.dz      = self.z[1]-self.z[0]
            self.binlow  = self.z-self.dz/2.
            self.binhigh = self.z+self.dz/2.
        else:
            self.binlow  = np.array(self.params['pdf_z'][:-1])
            self.binhigh = np.array(self.params['pdf_z'][1:])
            self.z       = (self.binlow+self.binhigh)/2.
            self.dz      = self.binlow[1]-self.binlow[0]

        print '\n\npassed first part\n\n'

        if hasattr(self.params['zbins'], "__len__"):
            self.tomobins = len(self.params['zbins']) - 1
            self.binedges = self.params['zbins']
        else:
            self.tomobins = self.params['zbins']
            #self.binedges = self.find_bin_edges(self.pz['pzbin'][self.mask], self.tomobins, w = self.shape['weight'][self.mask]) #Lucas: original
            self.binedges = self.find_bin_edges(self.selector_pz.get_col(self.Dict.pz_dict['pzbin']), self.tomobins, w = self.shape['weight'][self.mask])
            print 'remind troxel to replace this function to use destest classes' #troxel: don't remove
            
        if self.params['lensfile'] != 'None':
            if hasattr(self.params['lens_zbins'], "__len__"):
                self.lens_tomobins = len(self.params['lens_zbins']) - 1
                self.lens_binedges = self.params['lens_zbins']
            else:
                self.lens_tomobins = self.params['lens_zbins']
                #self.lens_binedges = self.find_bin_edges(self.lens_pz['pzbin'], self.lens_tomobins, w = self.lens['weight']) #Lucas: original
                self.lens_binedges = self.find_bin_edges(self.lens_pz['pzbin'], self.lens_tomobins, w = self.lens['weight']) 
                #deal with lenses later
        print 'passed second part\n\n'
        
        return

    def run(self):
        
        # Calculate source n(z)s and write to file
        pzbin = self.selector_pz.get_col(self.Dict.pz_dict['pzbin'])
        print 'passed third part\n\n'
        
        if self.params['pdf_type']!='pdf': #look at function build_nofz_bins        
            zbin, self.nofz = self.build_nofz_bins(
                self.tomobins,#created in init
                self.binedges,#same
                pzbin,
                self.selector_pz.get_col(self.Dict.pz_dict['pzstack'])[self.Dict.ind['u']],
                self.params['pdf_type'],
                self.weight,
                shape=True)
        else: #I don't know what happens if you fall here. Certainly the pipeline will fail
            print '\nThe pipeline will certainly fail now...\n'
            pdfs = np.zeros((len(self.pz),len(self.z)))
            for i in range(len(self.z)):
                pdfs[:,i] = self.pz['pzstack'+str(i)]
            zbin, self.nofz = self.build_nofz_bins(
                               self.tomobins,
                               self.binedges,
                               pzbin,
                               pdfs,
                               self.params['pdf_type'],
                               self.weight,
                               shape=True)

        print '\n\n passed fourth part\n\n '

        self.get_sige_neff(zbin,self.tomobins)

        f = h5py.File( self.output_path("nz_source"), mode='w')
        for zbin_,zname in tuple(zip(zbin,['zbin','zbin_1p','zbin_1m','zbin_2p','zbin_2m'])):
            print 'zbin_,zname=',zbin_,zname
            f.create_dataset( 'nofz/'+zname, maxshape=(len(self.selector_mcal.mask_),), shape=(len(zbin_),), dtype=zbin.dtype, chunks=(1000000,) )
            f['nofz/'+zname] = zbin_
        f.close()

        print '\n\n passed fifth part\n\n '

        # Calculate lens n(z)s and write to file
        lens_pzbin = self.selector_lens.get_col(self.Dict.lens_pz_dict['pzbin'])
        lens_pzstack = self.selector_lens.get_col(self.Dict.lens_pz_dict['pzstack'])
        lens_weight = self.calibrator_lens.calibrate(self.Dict.lens_pz_dict['weight'],weight_only=True) 
                
        if self.params['lensfile'] != 'None':
            lens_zbin, self.lens_nofz = self.build_nofz_bins(
                                         self.lens_tomobins,
                                         self.lens_binedges,
                                         lens_pzbin,
                                         lens_pzstack,
                                         self.params['lens_pdf_type'],
                                         lens_weight)
            print '\n\nsaving...\n\n'

            f = h5py.File( self.output_path("nz_lens"), mode='r+')
            f.create_dataset( 'nofz/lens_zbin', maxshape=(len(lens_zbin),), shape=(len(lens_zbin),), dtype=self.lens_zbin.dtype, chunks=(1000000,) )
            f['nofz/lens_zbin'] = lens_zbin

            ran_binning = np.digitize(self.selector_random.get_col(self.Dict.ran_dict['ranbincol']), self.lens_binedges, right=True) - 1
            f.create_dataset( 'nofz/ran_zbin', maxshape=(len(ran_binning),), shape=(len(ran_binning),), dtype=ran_binning.dtype, chunks=(1000000,) )
            f['nofz/ran_zbin'] = ran_binning

            f.close()

            self.get_lens_neff(lens_zbin,self.lens_tomobins)

    def write(self):
        """
        Write lens and source n(z)s to fits file for tomographic and non-tomographic cases.
        """

        nz_source = twopoint.NumberDensity(
                     NOFZ_NAMES[0],
                     self.binlow, 
                     self.z, 
                     self.binhigh, 
                     [self.nofz[i,:] for i in range(self.tomobins)])

        nz_source.ngal      = self.neff
        nz_source.sigma_e   = self.sigma_e
        nz_source.area      = self.area
        kernels             = [nz_source]
        np.savetxt(self.output_path("nz_source_txt"), np.vstack((self.binlow, self.nofz)).T)

        if self.params['lensfile'] != 'None':
            nz_lens      = twopoint.NumberDensity(
                            NOFZ_NAMES[1], 
                            self.binlow, 
                            self.z, 
                            self.binhigh, 
                            [self.lens_nofz[i,:] for i in range(self.lens_tomobins)])
            nz_lens.ngal = self.lens_neff
            nz_lens.area = self.area
            kernels.append(nz_lens)
            np.savetxt(self.output_path("nz_lens_txt"), np.vstack((self.binlow, self.lens_nofz)).T)

        data             = twopoint.TwoPointFile([], kernels, None, None)
        data.to_fits(self.output_path("2pt"), clobber=True)

        self.write_metadata()

    def write_metadata(self):
        import yaml
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
        if 'pzbin_col' in self.gold.dtype.names:
            data["source_bins"] = "gold_file_bins"
        else:
            if type(self.binedges) is list:
                data.update({ "source_bins" : self.binedges })
            else:
                data.update({ "source_bins" : self.binedges.tolist() })

        if self.params['lensfile'] != 'None':
            data.update({ "lens_neff" : self.lens_neff,
                          "lens_tomobins" : self.lens_tomobins,
                          "lens_bins" : self.lens_binedges })
        print data
        filename = self.output_path('metadata')
        open(filename, 'w').write(yaml.dump(data))


    # def load_array(self,d,file):


    #     if self.params['has_sheared'] & (file=='shapefile'):
    #         d['flags_1p'] = 'flags_select_1p'
    #         d['flags_1m'] = 'flags_select_1m'
    #         d['flags_2p'] = 'flags_select_2p'
    #         d['flags_2m'] = 'flags_select_2m'

    #     if self.params['pdf_type']=='pdf':
    #         keys = [key for key in d.keys() if (d[key] is not None)&(key is not 'pzstack')]
    #     else:
    #         keys = [key for key in d.keys() if (d[key] is not None)]

    #     if 'objid' in keys:
    #         dtypes = [('objid','i8')]
    #     else:
    #         raise ValueError('missing object id in '+file)
    #     dtypes += [(key,'f8') for key in keys if (key is not 'objid')]
    #     if self.params['pdf_type']=='pdf':
    #         dtypes  += [('pzstack_'+str(i),'f8') for i in range(len(self.params['pdf_z']))]

    #     fits = fio.FITS(self.params[file])[-1]
    #     array = fits.read(columns=[d[key] for key in keys]) #Lucas: somewhere around here?

    #     array = rename_fields(array,{v: k for k, v in d.iteritems()})

    #     if ('weight' not in array.dtype.names) & (file=='shapefile'):
    #         array = append_fields(array, 'weight', np.ones(len(array)), usemask=False)

    #     if self.params['pdf_type']=='pdf':
    #         for i in range(len(self.params['pdf_z'])):
    #             array['pzstack'+str(i)]=fits.read(columns=d['pzstack']+str(i))

    #     if np.any(np.diff(array['objid']) < 1):
    #         raise ValueError('misordered or duplicate ids in '+file) 

    #     return array

    # def load_data(self): #Lucas: modify here to read hdf5 files, modify also measure2pt.py
    #     """
    #     Load data files.
    #     """

    #     import time
    #     import importlib
    #     col = importlib.import_module('.'+self.params['dict_file'],'pipeline')

    #     # def lowercase_array(arr):
    #     #     old_names = arr.dtype.names
    #     #     new_names = [name.lower() for name in old_names]
    #     #     renames   = dict(zip(old_names, new_names))
    #     #     return rename_fields(arr, renames)
    #     t0 = time.time()
        
        
    #     self.gold      = self.load_array(col.gold_dict, 'goldfile')
    #     print 'Done goldfile',time.time()-t0,self.gold.dtype.names
        
    #     #print self.gold #Lucas: test
    #     #print destest.Selector.get_col('goldfile') #Lucas: test

    #     self.shape     = self.load_array(col.shape_dict, 'shapefile')
    #     print 'Done shapefile',time.time()-t0,self.shape.dtype.names
    #     self.pz        = self.load_array(col.pz_bin_dict, 'photozfile')
    #     print 'Done pzfile',time.time()-t0,self.pz.dtype.names
    #     if self.params['has_sheared']:
    #         self.pz_1p = self.load_array(col.pz_bin_dict, 'photozfile_1p')
    #         print 'Done pz1pfile',time.time()-t0,self.pz_1p.dtype.names
    #         self.pz_1m = self.load_array(col.pz_bin_dict, 'photozfile_1m')
    #         print 'Done pz1mfile',time.time()-t0,self.pz_1m.dtype.names
    #         self.pz_2p = self.load_array(col.pz_bin_dict, 'photozfile_2p')
    #         print 'Done pz2pfile',time.time()-t0,self.pz_2p.dtype.names
    #         self.pz_2m = self.load_array(col.pz_bin_dict, 'photozfile_2m')
    #         print 'Done pz2mfile',time.time()-t0,self.pz_2m.dtype.names
    #     self.pz_nofz   = self.load_array(col.pz_stack_dict, 'photozfile_nz')
    #     print 'Done pznofzfile',time.time()-t0,self.pz_nofz.dtype.names
    #     if self.params['lensfile'] != 'None':
    #         self.lens      = self.load_array(col.lens_dict, 'lensfile')
    #         print 'Done lensfile',time.time()-t0,self.lens.dtype.names
    #         self.lens_pz   = self.load_array(col.lens_pz_dict, 'lensfile')
    #         print 'Done lens_pzfile',time.time()-t0,self.lens_pz.dtype.names

    #     if 'm1' not in self.shape.dtype.names:
    #         self.shape = append_fields(self.shape, 'm1', self.shape['m2'], usemask=False)
    #     if 'm2' not in self.shape.dtype.names:
    #         self.shape = append_fields(self.shape, 'm2', self.shape['m1'], usemask=False)
    #     if self.params['oneplusm']==False:
    #         print 'converting m to 1+m'
    #         self.shape['m1'] = np.copy(self.shape['m1'])+1.
    #         self.shape['m2'] = np.copy(self.shape['m2'])+1.
    #     if 'c1' in self.shape.dtype.names:
    #         self.shape['e1'] -= self.shape['c1']
    #         self.shape['e2'] -= self.shape['c2']
    #         self.shape['c1'] = None
    #         self.shape['c2'] = None
    #     if self.params['flip_e2']==True:
    #         print 'flipping e2'
    #         self.shape['e2']*=-1
    #     if 'pzbin' not in self.lens_pz.dtype.names:
    #         self.lens_pz = append_fields(self.lens_pz, 'pzbin', self.lens_pz['pzstack'], usemask=False)
    #     if 'pzstack' not in self.lens_pz.dtype.names:
    #         self.lens_pz = append_fields(self.lens_pz, 'pzstack', self.lens_pz['pzbin'], usemask=False)

    #     if not ((len(self.gold)==len(self.shape))
    #         & (len(self.gold)==len(self.pz))
    #         & (len(self.gold)==len(self.pz_nofz))):
    #         raise ValueError('shape, gold, or photoz length mismatch')
    #     if self.params['has_sheared']:
    #         if not ((len(self.gold)==len(self.pz_1p))
    #             & (len(self.gold)==len(self.pz_1m))
    #             & (len(self.gold)==len(self.pz_2p))
    #             & (len(self.gold)==len(self.pz_2m))):
    #             raise ValueError('shape, gold, or photoz length mismatch')        
    #     if self.params['lensfile'] != 'None':
    #         if (len(self.lens)!=len(self.lens_pz)):
    #             raise ValueError('lens and lens_pz length mismatch') 

    #     if self.params['lensfile'] != 'None':
    #         keys = [key for key in col.ran_dict.keys() if (col.ran_dict[key] is not None)]
    #         fits = fio.FITS(self.params['randomfile'])[-1]

    #         dtypes=[(key,'f8') for key in keys]
    #         self.randoms = np.empty(fits.read_header()['NAXIS2'], dtype = dtypes)
    #         for key in keys:
    #             self.randoms[key]=fits.read(columns=[col.ran_dict[key]])

    #     if self.params['test_run']==True:
    #         idx = np.random.choice(np.arange(len(self.gold)),100000,replace=False)
    #         np.save(self.output_path("gold_idx"), idx)
    #         self.gold    = self.gold[idx]
    #         self.shape   = self.shape[idx]
    #         self.pz      = self.pz[idx]
    #         self.pz_nofz = self.pz_nofz[idx]
    #         if self.params['has_sheared']:
    #             self.pz_1p   = self.pz_1p[idx]
    #             self.pz_1m   = self.pz_1m[idx]
    #             self.pz_2p   = self.pz_2p[idx]
    #             self.pz_2m   = self.pz_2m[idx]
    #         if self.params['lensfile'] != 'None':
    #             idx = np.random.choice(np.arange(len(self.lens)),100000,replace=False)
    #             np.save(self.output_path("lens_idx"), idx)
    #             self.lens    = self.lens[idx]
    #             self.lens_pz = self.lens_pz[idx]
    #             idx = np.random.choice(np.arange(len(self.randoms)),100000,replace=False)
    #             np.save(self.output_path("ran_idx"), idx)
    #             self.randoms = self.randoms[idx]

    #     if 'pzbin_col' in self.gold.dtype.names:
    #         mask = (self.gold['pzbin_col'] >= 0)
    #     else:
    #         mask = (self.pz['pzbin'] > self.params['zlims'][0]) & (self.pz['pzbin'] <= self.params['zlims'][1])
    #         if self.params['has_sheared']:
    #             mask_1p = (self.pz_1p['pzbin'] > self.params['zlims'][0]) & (self.pz_1p['pzbin'] <= self.params['zlims'][1])
    #             mask_1m = (self.pz_1m['pzbin'] > self.params['zlims'][0]) & (self.pz_1m['pzbin'] <= self.params['zlims'][1])
    #             mask_2p = (self.pz_2p['pzbin'] > self.params['zlims'][0]) & (self.pz_2p['pzbin'] <= self.params['zlims'][1])
    #             mask_2m = (self.pz_2m['pzbin'] > self.params['zlims'][0]) & (self.pz_2m['pzbin'] <= self.params['zlims'][1])

    #     print 'ngal',np.sum(mask),np.sum(mask_1p),np.sum(mask_1m),np.sum(mask_2p),np.sum(mask_2m)
    #     if 'flags' in self.shape.dtype.names:
    #         mask = mask & (self.shape['flags']==0)
    #         if self.params['has_sheared']:
    #             mask_1p = mask_1p & (self.shape['flags_1p']==0)
    #             mask_1m = mask_1m & (self.shape['flags_1m']==0)
    #             mask_2p = mask_2p & (self.shape['flags_2p']==0)
    #             mask_2m = mask_2m & (self.shape['flags_2m']==0)
    #     print 'ngal',np.sum(mask),np.sum(mask_1p),np.sum(mask_1m),np.sum(mask_2p),np.sum(mask_2m)
    #     if 'flags' in self.pz.dtype.names:
    #         mask = mask & (self.pz['flags']==0)
    #         if self.params['has_sheared']:
    #             mask_1p = mask_1p & (self.pz_1p['flags']==0)
    #             mask_1m = mask_1m & (self.pz_1m['flags']==0)
    #             mask_2p = mask_2p & (self.pz_2p['flags']==0)
    #             mask_2m = mask_2m & (self.pz_2m['flags']==0)

    #     print 'hardcoded spt region cut'
    #     mask = mask & (self.shape['dec']<-35)
    #     if self.params['has_sheared']:
    #         mask_1p = mask_1p & (self.shape['dec']<-35)
    #         mask_1m = mask_1m & (self.shape['dec']<-35)
    #         mask_2p = mask_2p & (self.shape['dec']<-35)
    #         mask_2m = mask_2m & (self.shape['dec']<-35)

    #     np.save('radec.npy',np.vstack((self.shape['ra'],self.shape['dec'])).T[mask])

    #     print np.sum(mask)
    #     if 'footprintfile' in self.params.keys():
    #         print 'cutting catalog to footprintfile'
    #         footmask = np.in1d(hp.ang2pix(4096, np.pi/2.-np.radians(self.shape['dec']),np.radians(self.shape['ra']), nest=False),
    #                            fio.FITS(self.params['footprintfile'])[-1].read()['HPIX'],assume_unique=False)
    #         mask = mask & footmask
    #         if self.params['has_sheared']:
    #             mask_1p = mask_1p & footmask
    #             mask_1m = mask_1m & footmask
    #             mask_2p = mask_2p & footmask
    #             mask_2m = mask_2m & footmask

    #     print 'ngal final',np.sum(mask),np.sum(mask & mask_1p & mask_1m & mask_2p & mask_2m)

    #     if self.params['has_sheared']:
    #         full_mask     = mask | mask_1p | mask_1m | mask_2p | mask_2m
    #         self.pz_1p    = self.pz_1p[full_mask]
    #         self.pz_1m    = self.pz_1m[full_mask]
    #         self.pz_2p    = self.pz_2p[full_mask]
    #         self.pz_2m    = self.pz_2m[full_mask]
    #         self.mask     = mask[full_mask]
    #         self.mask_1p  = mask_1p[full_mask]
    #         self.mask_1m  = mask_1m[full_mask]
    #         self.mask_2p  = mask_2p[full_mask]
    #         self.mask_2m  = mask_2m[full_mask]
    #     else:
    #         full_mask  = mask
    #         self.mask  = mask[full_mask]
    #     self.gold      = self.gold[full_mask]
    #     self.shape     = self.shape[full_mask]
    #     self.pz        = self.pz[full_mask]
    #     self.pz_nofz   = self.pz_nofz[full_mask]

    #     return 

    def build_nofz_bins(self, zbins, edge, bin_col, stack_col, pdf_type, weight,shape=False):
        """
        Build an n(z), non-tomographic [:,0] and tomographic [:,1:].
        """

        #R,c,w = self.calibrator.calibrate('e1',mask=[mask,mask_1p,mask_1m,mask_2p,mask_2m]) #Lucas: attempting to load mask here.
        if shape&(self.params['has_sheared']):
            #if 'pzbin_col' in self.gold.dtype.names:
            #    xbins = self.gold['pzbin_col']
            #else:
            xbins0=[]
            for x in bin_col:
                xbins0.append(np.digitize(x, edge, right=True) - 1)
            xbins = xbins0[0]
        else:
            #if 'pzbin_col' in self.gold.dtype.names:
            #    xbins0 = self.gold['pzbin_col']
            #else:
            xbins0 = np.digitize(bin_col, edge, right=True) - 1
            xbins=xbins0

        # Stack n(z)
        nofz  = np.zeros((zbins, len(self.z)))

        # MC Sample of pdf or redmagic (if redmagic, takes random draw from gaussian of width 0.01)
        if (pdf_type == 'sample') | (pdf_type == 'rm'):
            if pdf_type == 'rm':
                #stack_col = np.random.normal(stack_col, self.lens_pz['pzerr']*np.ones(len(stack_col)))
                stack_col = np.random.normal(stack_col, self.selector_lens.get_col(self.Dict.lens_pz_dict['pzerr'])*np.ones(len(stack_col)))
            for i in range(zbins):
                mask        =  (xbins == i)
                if shape:
                    #mask = mask&self.mask #Lucas: forget this mask since get_col deals with it
                    if self.params['has_sheared']:
                        
                        mask_1p = (xbins0[1] == i)
                        mask_1m = (xbins0[2] == i)
                        mask_2p = (xbins0[3] == i)
                        mask_2m = (xbins0[4] == i)

                        if len(weight)<=5:
                            weight_ = weight[0]*self.calibrator.calibrate('e1',mask=[mask],return_wRg=True) # This returns an array of (Rg1+Rg2)/2*w for weighting the n(z)
                        else:
                            weight_ = weight*self.calibrator.calibrate('e1',mask=[mask],return_wRg=True) # This returns an array of (Rg1+Rg2)/2*w for weighting the n(z) 
                        print 'check that theres no double weighting in final pipeline' #troxel: don't remove
                        
                    else:
                        m1 = self.shape['m1']
                        m2 = self.shape['m2']
                        weight_ = weight*(m1+m2)/2.
                else:
                    weight_ = weight
                if np.isscalar(weight_):
                    nofz[i,:],b =  np.histogram(stack_col[mask], bins=np.append(self.binlow, self.binhigh[-1]))
                else:
                    nofz[i,:],b =  np.histogram(stack_col[mask], bins=np.append(self.binlow, self.binhigh[-1]), weights=weight_)
                nofz[i,:]   /= np.sum(nofz[i,:]) * self.dz

        # Stacking pdfs
        elif pdf_type == 'pdf':
            for i in xrange(zbins):
                mask      =  (xbins == i)
                if shape:
                    mask = mask&self.mask
                nofz[i,:] =  np.sum((stack_col[mask].T * weight[mask]).T, axis=0)
                nofz[i,:] /= np.sum(nofz[i,:]) * self.dz

        return xbins0, nofz

    def find_bin_edges(self, x, nbins, w=None):
        """
        For an array x, returns the boundaries of nbins equal (possibly weighted by w) bins.
        From github.com/matroxel/destest.
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

    def get_lens_neff(self, zbin, tomobins):
        """
        Calculate neff for catalog.
        """

        if not hasattr(self,'area'):
            self.get_area()

        self.lens_neff = []
        for i in range(tomobins):
          mask = (zbin == i)
          a    = np.sum(cat['weight'][mask])**2
          b    = np.sum(cat['weight'][mask]**2)
          c    = self.area * 60. * 60.

          self.lens_neff.append(np.asscalar( a/b/c ))

        return


    def get_sige_neff(self, zbin, tomobins):

        if not hasattr(self,'area'):
            self.get_area()

        self.mean_e1 = []
        self.mean_e2 = []
        self.sigma_e = []
        self.sigma_ec = []
        self.neff = []
        self.neffc = []
        e1_  = self.selector_mcal.get_col(self.Dict.shape_dict['e1'],nosheared=True)[0]
        e2_  = self.selector_mcal.get_col(self.Dict.shape_dict['e2'],nosheared=True)[0]
        cov00_  = self.selector_mcal.get_col(self.Dict.shape_dict['cov00'],nosheared=True)[0]
        cov11_  = self.selector_mcal.get_col(self.Dict.shape_dict['cov11'],nosheared=True)[0]
        for i in range(tomobins):
            if self.params['has_sheared']:
                mask = (zbin[0] == i)
                mask_1p = (zbin[1] == i)
                mask_1m = (zbin[2] == i)
                mask_2p = (zbin[3] == i)
                mask_2m = (zbin[4] == i)

                R,c,w = self.calibrator.calibrate('e1',mask=[mask,mask_1p,mask_1m,mask_2p,mask_2m]) #Added by Troxel. Lucas: R will be the final mean response

            else:
                mask = (zbin == i)
                m1 = cat['m1']
                m2 = cat['m2']

            if self.params['has_sheared']:
                e1  = e1_[mask]
                e2  = e2_[mask]
                s   = R
                var = cov00_[mask]+cov11_[mask]
                var[var>2] = 2.
            else:
                e1  = cat['e1'][mask]
                e2  = cat['e2'][mask]
                w   = cat['weight'][mask]
                s = (m1[mask]+m2[mask])/2.
                snvar = 0.24#np.sqrt((cat['e1'][mask][cat['snr'][mask]>100].var()+cat['e2'][mask][cat['snr'][mask]>100].var())/2.)
                print i,'snvar',snvar
                var = 1./w - snvar**2
                var[var < 0.] = 0.
                w[w > snvar**-2] = snvar**-2
                print 'var',var.min(),var.max()

            if np.isscalar(w):
                self.mean_e1.append(np.asscalar(np.mean(e1))) # this is without calibration factor!
                self.mean_e2.append(np.asscalar(np.mean(e2)))
            else:
                self.mean_e1.append(np.asscalar(np.average(e1,weights=w[0]))) # this is without calibration factor!
                self.mean_e2.append(np.asscalar(np.average(e2,weights=w[0])))

            a1 = np.sum(w**2 * (e1-self.mean_e1[i])**2)
            a2 = np.sum(w**2 * (e2-self.mean_e2[i])**2)
            b  = np.sum(w**2)
            c  = np.sum(w * s)
            d  = np.sum(w)

            self.sigma_e.append( np.sqrt( (a1/c**2 + a2/c**2) * (d**2/b) / 2. ) )
            self.sigma_ec.append( np.sqrt( np.sum(w**2 * (e1**2 + e2**2 - var)) / (2.*np.sum(w**2 * s**2)) ) )

            a    = np.sum(w)**2
            c    = self.area * 60. * 60.

            self.neff.append( a/b/c )
            self.neffc.append( ((self.sigma_ec[i]**2 * np.sum(w * s)**2) / np.sum(w**2 * (s**2 * self.sigma_ec[i]**2 + var/2.))) / self.area / 60**2 )

    def get_area(self):

        if hasattr(self,'area'):
            return

        if self.params['area']=='None':

            import healpy as hp

            pix=hp.ang2pix(4096, np.pi/2.-np.radians(self.shape['dec']),np.radians(self.shape['ra']), nest=True)
            area=hp.nside2pixarea(4096)*(180./np.pi)**2
            mask=np.bincount(pix)>0
            self.area=np.sum(mask)*area
            self.area=float(self.area)
            print self.area
        
        else:

            self.area = self.params['area']

        return 

def find_git_hash():
    try:
        dirname = os.path.dirname(os.path.abspath(__file__))
        head = subprocess.check_output("cd {0}; git show-ref HEADS".format(dirname), shell=True)
    except subprocess.CalledProcessError:
        head = "UNKNOWN"
        warnings.warn("Unable to find git repository commit ID in {}".format(dirname))
    return head.split()[0]


