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

        #print 'passed first part\n\n'

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
        print 'Binning was set up'
        
        return

    def run(self):
        
        # Calculate source n(z)s and write to file
        print 'before pzbin ---------------------'
        pzbin = self.selector_pz.get_col(self.Dict.pz_dict['pzbin'])
        print 'before e1 ---------------------'
        e1 = self.selector_mcal.get_col('e1')
        print 'In run: pzbin = ',len(pzbin[0]),pzbin

        if self.params['pdf_type']!='pdf': 
            zbin, self.nofz = self.build_nofz_bins(
                self.tomobins,
                self.binedges,
                pzbin,
                self.selector_pz.get_col(self.Dict.pz_dict['pzstack'])[self.Dict.ind['u']],
                self.params['pdf_type'],
                self.weight,
                shape=True)
        else: 
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

        print '\nCalculated source n(z), now getting sigma_e and Neff '

        self.get_sige_neff(zbin,self.tomobins)

        f = h5py.File( self.output_path("nz_source"), mode='w')
        for zbin_,zname in tuple(zip(zbin,['zbin','zbin_1p','zbin_1m','zbin_2p','zbin_2m'])):
            f.create_dataset( 'nofz/'+zname, maxshape=(2*len(zbin_),), shape=(len(zbin_),), dtype=zbin_.dtype, chunks=(len(zbin_)/10,) )
            f['nofz/'+zname][:] = zbin_
        f.close()

        print 'Calculated sigma_e and Neff for sources.\nCalculating lens n(z)'

        # Calculate lens n(z)s and write to file
        lens_pzbin = self.selector_lens.get_col(self.Dict.lens_pz_dict['pzbin'])[0]
        lens_pzstack = self.selector_lens.get_col(self.Dict.lens_pz_dict['pzstack'])[0]
        lens_weight = self.calibrator_lens.calibrate(self.Dict.lens_pz_dict['weight'],weight_only=True) 
                
        if self.params['lensfile'] != 'None':
            lens_zbin, self.lens_nofz = self.build_nofz_bins(
                                         self.lens_tomobins,
                                         self.lens_binedges,
                                         lens_pzbin,
                                         lens_pzstack,
                                         self.params['lens_pdf_type'],
                                         lens_weight)
            print 'Saving lens n(z)',len(lens_zbin),len(lens_pzbin)

            f = h5py.File( self.output_path("nz_source"), mode='r+')
            f.create_dataset( 'nofz/lens_zbin', maxshape=(len(lens_zbin),), shape=(len(lens_zbin),), dtype=lens_zbin.dtype, chunks=(len(lens_zbin)/10,) )
            f['nofz/lens_zbin'][:] = lens_zbin

            ran_binning = np.digitize(self.selector_random.get_col(self.Dict.ran_dict['ranbincol'])[0], self.lens_binedges, right=True) - 1
            f.create_dataset( 'nofz/ran_zbin', maxshape=(len(ran_binning),), shape=(len(ran_binning),), dtype=ran_binning.dtype, chunks=(len(ran_binning)/10,) )
            f['nofz/ran_zbin'][:] = ran_binning

            f.close()

            if np.isscalar(lens_weight):
                self.get_lens_neff(lens_zbin,self.lens_tomobins,np.ones(len(lens_zbin)))
            else:
                self.get_lens_neff(lens_zbin,self.lens_tomobins,lens_weight)

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
        #if 'pzbin_col' in self.gold.dtype.names:
        #    data["source_bins"] = "gold_file_bins"
        #else:
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


    def build_nofz_bins(self, zbins, edge, bin_col, stack_col, pdf_type, weight,shape=False):
        """
        Build an n(z), non-tomographic [:,0] and tomographic [:,1:].
        """

        print ' ---- build nofz bins ',bin_col,np.min(bin_col),np.max(bin_col),stack_col

        #R,c,w = self.calibrator.calibrate('e1',mask=[mask,mask_1p,mask_1m,mask_2p,mask_2m]) #Lucas: attempting to load mask here.
        if shape&(self.params['has_sheared']):
            #if 'pzbin_col' in self.gold.dtype.names:
            #    xbins = self.gold['pzbin_col']
            #else:
            xbins0=[]
            print 'In build_nofz_bins: bin_col=',bin_col
            for x in bin_col:
                #print 'In build_nofz_bins: np.digitize(x, edge, right=True) - 1 = ',np.digitize(x, edge, right=True) - 1
                xbins0.append(np.digitize(x, edge, right=True) - 1)
                print np.min(x),np.max(x),np.min(xbins0[-1]),np.max(xbins0[-1])
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
                stack_col = np.random.normal(stack_col, self.selector_lens.get_col(self.Dict.lens_pz_dict['pzerr'])[0])
            for i in range(zbins):
                mask        =  (xbins == i)
                if shape:
                    #mask = mask&self.mask #Lucas: forget this mask since get_col deals with it
                    if self.params['has_sheared']:
                        #print 'In build_nofz_bins: xbins0=',xbins0,'\n\nI will crash now\n\n'
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

    def get_lens_neff(self, zbin, tomobins, weight):
        """
        Calculate neff for catalog.
        """

        if not hasattr(self,'area'):
            self.get_area()

        self.lens_neff = []
        for i in range(tomobins):
            print '\nDoing lens zbin',i
            

            mask = (zbin == i)
            a    = np.sum(weight[mask])**2
            b    = np.sum(weight[mask]**2)
            c    = self.area * 60. * 60.
            #print 'mask=',mask
            print np.sum(weight[mask]),'objects found in this bin',weight[mask]
            #print 'np.sum(weight)=',np.sum(weight)
            #print 'self.area=',self.area
            #print 'a=',a
            #print 'b=',b
            #print 'c=',c
            
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
            print '\nDoing source zbin',i
            if self.params['has_sheared']:
                mask = (zbin[0] == i)
                mask_1p = (zbin[1] == i)
                mask_1m = (zbin[2] == i)
                mask_2p = (zbin[3] == i)
                mask_2m = (zbin[4] == i)

                R,c,w = self.calibrator.calibrate('e1',mask=[mask,mask_1p,mask_1m,mask_2p,mask_2m]) #Added by Troxel. Lucas: R will be the final mean response
                if type(w) is list:
                    w = w[0]
                if np.isscalar(w):
                    print 'Re-defining w as np.ones(np.sum(mask))'
                    w = np.ones(np.sum(mask))

                print np.sum(mask),'objects found in this bin'
                e1  = e1_[mask]
                e2  = e2_[mask]
                s   = R
                var = cov00_[mask]+cov11_[mask]
                var[var>2] = 2.
            

            else:
                mask = (zbin == i)
                m1 = cat['m1']
                m2 = cat['m2']
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

            self.mean_e1.append(np.asscalar(np.average(e1,weights=w))) # this is without calibration factor!
            self.mean_e2.append(np.asscalar(np.average(e2,weights=w)))
            
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


