from __future__ import print_function, division
import h5py
import numpy as np
import twopoint
import fitsio as fio
import glob
import yaml
import math
import os
import collections
from . import blind_2pt_usingcosmosis as blind
import pickle
from .stage import PipelineStage, TWO_POINT_NAMES, NOFZ_NAMES

print(TWO_POINT_NAMES)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)#, encoding='latin1')

print('Imported modules')

class WriteFits(PipelineStage):
    name = "2pt_fits"
    inputs = {
        "ggl_bins"              : ("cov", "run.gglensing_zbins"),
        "cov"                   : ("cov", "cov.txt"),
        "covfiles"              : ("cov", "run_*"),

        "2pt"                   : ("nofz","2pt.fits"),
        "xipm"                  : ("2pt", "shear_shear"),
        "gammat"                : ("2pt", "shear_pos"),
        "wtheta"                : ("2pt", "pos_pos"),
        "nofz_meta"             : ("nofz", "metadata.yaml"),
    }
    outputs = {
        "2pt_extended"          : "2pt_extended_data.fits",
        "2pt_ng"                : "2pt_NG.fits",
        "2pt_g"                 : "2pt_G.fits",
    }

    def __init__(self,param_file):
        """
        Initialise object.
v        """
        super(WriteFits,self).__init__(param_file)

    def run(self):

        # Initialise twopoint spectrum classes
        self.init_specs()

        # Load xi data
        self.load_metadata()
        self.load_twopt_data()

        # Load covariance info
        self.load_cov()

        if 'no_blinding' in self.params:
            if self.params['no_blinding']:
                print("Not blinding")
                return
        print('\nBlinding will be applied to the file being written\nThis is hard-coded.')
        print('To unblind, set do_Blinding=False in write_fits.py')
        self.blind()

        return

    def blind(self):

        import os

        os.system('bash pipeline/BASHTEST.sh')

        return

    def strip_wtheta(self, fits):
        if self.params['cross_clustering']:
            return
        # AA: no idea what this is for?

        wtheta = fits.get_spectrum(TWO_POINT_NAMES[-1])
        print(wtheta.bin1, wtheta.bin2)

        mask = wtheta.bin1==wtheta.bin2
        print("Cutting out {} values from wtheta because no cross_clustering".format(len(mask)-mask.sum()))
        wtheta.apply_mask(mask)

    def strip_missing_gglensing(self, fits):

        filename = self.input_path("ggl_bins")
        bin1,bin2,accept,_=np.loadtxt(filename).T
        bin1,bin2,accept,_=np.loadtxt(filename).T
        bin1=bin1.astype(int) + 1
        bin2=bin2.astype(int) + 1
        accept=accept.astype(int)

        gammat = fits.get_spectrum(TWO_POINT_NAMES[2])

        accept_dict = {}
        for (b1,b2,a) in zip(bin1,bin2,accept):
            accept_dict[(b1,b2)] = a

        mask = [accept_dict[(b1,b2)] for (b1,b2) in zip(gammat.bin1,gammat.bin2)]
        mask = np.array(mask, dtype=bool)

        print("Cutting out {} values from gammat because lens behind source".format(len(mask)-mask.sum()))

        gammat.apply_mask(mask)


    def cut_gammax(self,fits):
        gammax = fits.get_spectrum(TWO_POINT_NAMES[3])
        print(gammax.bin1, gammax.bin2)


        print("Cutting out gammax vales")

    def write(self):

        # Load 2point fits file with n(z) info.
        fits=twopoint.TwoPointFile.from_fits(self.input_path("2pt"),covmat_name=None)

        # Write file without covariance (all data vectors)
        fits.spectra=self.exts
        fits.to_fits(self.output_path("2pt_extended"), clobber=True)

        if self.covmat is not None:
        #AA: I think these things should happen for g and ng, even if no cov
            self.strip_wtheta(fits)
            self.strip_missing_gglensing(fits)
            length=self.get_cov_lengths(fits)

        # self.sort_2pt(fits,length) # Now fixed sorting to match cosmolike

        # Writes the covariance info into a covariance object and saves to 2point fits file.
        if self.covmat is not None:
            fits.covmat_info=twopoint.CovarianceMatrixInfo('COVMAT',TWO_POINT_NAMES,length,self.covmat[0])
        fits.to_fits(self.output_path("2pt_g"),clobber=True)
        if self.covmat is not None:
            fits.covmat_info=twopoint.CovarianceMatrixInfo('COVMAT',TWO_POINT_NAMES,length,self.covmat[1])
        fits.to_fits(self.output_path("2pt_ng"),clobber=True)

        print("Have disabled covmat cleanup")
        # self.cleanup_cov()

    def load_metadata(self):
        """
        Read metadata from nofz stage.
        """
        filename = self.input_path('nofz_meta')
        data = yaml.unsafe_load(open(filename))
        self.mean_e1 = np.array(data['mean_e1'])
        self.mean_e2 = np.array(data['mean_e2'])
        self.zbins = len(np.array(data['source_bins'])) - 1
        self.lens_zbins = len(np.array(data['lens_bins'])) - 1

    def load_twopt_data(self):

        def get_length(n,n2=None):
            if n2 is not None:
                return n*n2
            else:
                return n*(n+1)/2

        nbins = self.params['tbins']
        min_sep, max_sep = self.params['tbounds']
        bin_size = math.log(max_sep / min_sep)/ nbins        
        logr = np.linspace(0, nbins*bin_size, nbins, endpoint=False,
                                   dtype=float)
        logr += math.log(min_sep) + 0.5*bin_size
        
        rnom = np.exp(logr)
        half_bin = np.exp(0.5*bin_size)
        left_edges = rnom / half_bin
        right_edges = rnom * half_bin

        # Cosmic shear
        if (self.params['region_mode'] == 'pixellized') or (self.params['region_mode'] == 'both'):
            f = load_obj(self.input_path("xipm")+'_pixellized')
            length = int(get_length(self.zbins)*self.params['tbins'])
            print("here")
            self.exts[0].angular_bin = np.zeros(length)
            self.exts[0].angle       = np.zeros(length)
            self.exts[0].angle_min   = np.zeros(length)
            self.exts[0].angle_max   = np.zeros(length)            
            self.exts[0].bin1        = np.zeros(length)
            self.exts[0].bin2        = np.zeros(length)
            self.exts[0].value       = np.zeros(length)
            self.exts[0].npairs      = np.zeros(length)
            self.exts[0].weight      = np.zeros(length)
            self.exts[1].angular_bin = np.zeros(length)
            self.exts[1].angle_min   = np.zeros(length)
            self.exts[1].angle_max   = np.zeros(length)                        
            self.exts[1].angle       = np.zeros(length)
            self.exts[1].bin1        = np.zeros(length)
            self.exts[1].bin2        = np.zeros(length)
            self.exts[1].value       = np.zeros(length)
            self.exts[1].npairs      = np.zeros(length)
            self.exts[1].weight      = np.zeros(length)
            for i,bins in enumerate(np.sort(list(f.keys()))):
                self.exts[0].bin1[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])]        = int(bins[0])+1
                self.exts[0].bin2[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])]        = int(bins[-1])+1
                self.exts[0].angular_bin[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])] = np.arange(int(self.params['tbins']))
                self.exts[0].angle_min[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])] = left_edges
                self.exts[0].angle_max[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])] = right_edges                
                self.exts[0].angle[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])]       = np.exp(f[bins]['meanlogr'])
                self.exts[0].value[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])]     = f[bins]['xip']
                self.exts[0].npairs[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])]      = f[bins]['npairs']
                self.exts[0].weight[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])]    = f[bins]['weight']
                self.exts[1].bin1[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])]        = int(bins[0])+1
                self.exts[1].bin2[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])]        = int(bins[-1])+1
                self.exts[1].angular_bin[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])] = np.arange(int(self.params['tbins']))
                self.exts[1].angle_min[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])] = left_edges
                self.exts[1].angle_max[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])] = right_edges
                self.exts[1].angle[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])]       = np.exp(f[bins]['meanlogr'])
                self.exts[1].value[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])]     = f[bins]['xim']
                self.exts[1].npairs[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])]      = f[bins]['npairs']
                self.exts[1].weight[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])]    = f[bins]['weight']

        # gammat
        if (self.params['region_mode'] == 'pixellized') or (self.params['region_mode'] == 'both'):
            f = load_obj(self.input_path("gammat")+'_pixellized')
            length = int(get_length(self.zbins,n2=self.lens_zbins)*self.params['tbins'])
            self.exts[2].angular_bin   = np.zeros(length)
            self.exts[2].angle_min     = np.zeros(length)
            self.exts[2].angle_max     = np.zeros(length)                        
            self.exts[2].angle         = np.zeros(length)
            self.exts[2].bin1          = np.zeros(length)
            self.exts[2].bin2          = np.zeros(length)
            self.exts[2].value         = np.zeros(length)
            self.exts[2].npairs        = np.zeros(length)
            self.exts[2].weight        = np.zeros(length)
            self.exts[2].random_npairs = np.zeros(length)
            self.exts[2].random_weight = np.zeros(length)
            self.exts[3].angular_bin   = np.zeros(length)
            self.exts[3].angle         = np.zeros(length)
            self.exts[3].angle_min     = np.zeros(length)
            self.exts[3].angle_max     = np.zeros(length)                        
            self.exts[3].bin1          = np.zeros(length)
            self.exts[3].bin2          = np.zeros(length)
            self.exts[3].value         = np.zeros(length)
            self.exts[3].npairs        = np.zeros(length)
            self.exts[3].weight        = np.zeros(length)
            self.exts[3].random_npairs = np.zeros(length)
            self.exts[3].random_weight = np.zeros(length)
            for i,bins in enumerate(np.sort(list(f.keys()))):
                self.exts[2].bin1[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])]        = int(bins[0])+1
                self.exts[2].bin2[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])]        = int(bins[-1])+1
                self.exts[2].angular_bin[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])] = np.arange(int(self.params['tbins']))
                self.exts[2].angle_min[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])] = left_edges
                self.exts[2].angle_max[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])] = right_edges
                self.exts[2].angle[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])]       = np.exp(f[bins]['meanlogr'])
                self.exts[2].value[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])]     = f[bins]['gammat_compens']
                self.exts[2].npairs[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])]      = f[bins]['npairs']
                self.exts[2].weight[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])]    = f[bins]['weight']
                self.exts[2].random_npairs[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])]   = f[bins]['npairs_rndm']
                self.exts[2].random_weight[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])] = f[bins]['weight_rndm']
                self.exts[3].bin1[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])]        = int(bins[0])+1
                self.exts[3].bin2[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])]        = int(bins[-1])+1
                self.exts[3].angular_bin[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])] = np.arange(int(self.params['tbins']))
                self.exts[3].angle_min[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])] = left_edges
                self.exts[3].angle_max[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])] = right_edges                
                self.exts[3].angle[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])]       = np.exp(f[bins]['meanlogr'])
                self.exts[3].value[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])]     = f[bins]['gammat_compens_im']
                self.exts[3].npairs[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])]      = f[bins]['npairs']
                self.exts[3].weight[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])]    = f[bins]['weight']
                self.exts[3].random_npairs[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])]   = f[bins]['npairs_rndm']
                self.exts[3].random_weight[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])] = f[bins]['weight_rndm']

        # galaxy clustering
        if (self.params['region_mode'] == 'pixellized') or (self.params['region_mode'] == 'both'):
            f = load_obj(self.input_path("wtheta")+'_pixellized')
            length = int(get_length(self.lens_zbins)*self.params['tbins'])
            self.exts[4].angular_bin   = np.zeros(length)
            self.exts[4].angle         = np.zeros(length)
            self.exts[4].angle_min     = np.zeros(length)
            self.exts[4].angle_max     = np.zeros(length)                        
            self.exts[4].bin1          = np.zeros(length)
            self.exts[4].bin2          = np.zeros(length)
            self.exts[4].value         = np.zeros(length)
            self.exts[4].npairs        = np.zeros(length)
            self.exts[4].weight        = np.zeros(length)
            # self.exts[4].random_npairs = np.zeros(length)
            # self.exts[4].random_weight = np.zeros(length)
            # self.exts[4].dr_npairs     = np.zeros(length)
            # self.exts[4].dr_weight     = np.zeros(length)
            # self.exts[4].rd_npairs     = np.zeros(length)
            # self.exts[4].rd_weight     = np.zeros(length)
            for i,bins in enumerate(np.sort(list(f.keys()))):
                self.exts[4].bin1[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])]        = int(bins[0])+1
                self.exts[4].bin2[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])]        = int(bins[-1])+1
                self.exts[4].angular_bin[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])] = np.arange(int(self.params['tbins']))
                self.exts[4].angle_min[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])] = left_edges
                self.exts[4].angle_max[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])] = right_edges                
                self.exts[4].angle[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])]       = np.exp(f[bins]['meanlogr'])
                self.exts[4].value[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])]     = f[bins]['w']
                self.exts[4].npairs[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])]      = f[bins]['npairs']
                self.exts[4].weight[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])]    = f[bins]['weight']
                # self.exts[4].random_npairs[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])]   = f[bins]['npairs_rndm']
                # self.exts[4].random_weight[[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])]] = f[bins]['weight_rndm']
                # self.exts[4].dr_npairs[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])]   = f[bins]['npairs_rndm']
                # self.exts[4].dr_weight[[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])]] = f[bins]['weight_rndm']
                # self.exts[4].rd_npairs[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])]   = f[bins]['npairs_rndm']
                # self.exts[4].rd_weight[[i*int(self.params['tbins']):(i+1)*int(self.params['tbins'])]] = f[bins]['weight_rndm']

    def load_cov(self):
        import os

        #if os.path.exists(self.input_path("cov")):
        #    os.remove(self.input_path("cov"))
        #os.system("cat "+self.input_path("covfiles")+" >> "+self.input_path("cov"))
        try:
            covdata = np.loadtxt(self.input_path("cov"))
        except:
            self.covmat = None
            print('Skipping covariance, since output file missing.')
            return

        # Replace theta values with bin numbers.
        theta=np.sort(np.unique(covdata[:,2]))
        for i in range(len(theta)):
          covdata[np.where(covdata[:,2]==theta[i])[0],2]=i
          covdata[np.where(covdata[:,3]==theta[i])[0],3]=i

        # Populate covariance matrix.
        ndata=int(np.max(covdata[:,0]))+1
        ndata2=int(np.max(covdata[:,1]))+1
        assert ndata==ndata2

        cov=np.zeros((ndata,ndata))
        cov[:,:] = 0.0
        for i in range(0,covdata.shape[0]):
            cov[int(covdata[i,0]),int(covdata[i,1])]=covdata[i,8]
            cov[int(covdata[i,1]),int(covdata[i,0])]=covdata[i,8]
        covNG=np.zeros((ndata,ndata))
        covNG[:,:] = 0.0
        for i in range(0,covdata.shape[0]):
            covNG[int(covdata[i,0]),int(covdata[i,1])]=covdata[i,8]+covdata[i,9]
            covNG[int(covdata[i,1]),int(covdata[i,0])]=covdata[i,8]+covdata[i,9]

        self.covmat = (cov,covNG)

        u,i           = np.unique(covdata[:,0],return_index=True)
        covdata       = covdata[i]
        self.covorder = np.vstack((covdata[:,2],covdata[:,4],covdata[:,5])).T

    def init_specs(self):

        NOFZ_NAMES=['nz_source','nz_lens']
        TWO_POINT_NAMES = ['xip','xim','gammat','gammax','wtheta']
        dtype1=[twopoint.Types.galaxy_shear_plus_real,twopoint.Types.galaxy_shear_minus_real,twopoint.Types.galaxy_position_real,twopoint.Types.galaxy_position_real,twopoint.Types.galaxy_position_real]
        dtype2=[twopoint.Types.galaxy_shear_plus_real,twopoint.Types.galaxy_shear_minus_real,twopoint.Types.galaxy_shear_plus_real,twopoint.Types.galaxy_position_real,twopoint.Types.galaxy_position_real]
        nznameindex1=[0,0,1,1,1]
        nznameindex2=[0,0,0,0,1]

        # Setup xi extensions
        self.exts=[]
        for i,name in enumerate(TWO_POINT_NAMES):
            self.exts.append(twopoint.SpectrumMeasurement(
                name, # hdu name
                ([],[]), # tomographic bins
                (dtype1[i], dtype2[i]), # type of 2pt statistic
                (NOFZ_NAMES[nznameindex1[i]], NOFZ_NAMES[nznameindex2[i]]), # associated nofz
                "SAMPLE", # window function
                None, # id
                None, # value
                npairs=None, # pair counts
                angle=None,
                angle_unit='arcmin')) # units

        return

    def get_cov_lengths(self,fits):

        # Calculate length of covariance blocks. Exception for gammat, which reads a file that stores which bin combinations have been rejected in the covariance calcultion.

        # Make cov lengths
        length=np.array([])
        for name in TWO_POINT_NAMES:
            twopt=fits.get_spectrum(name)
            if twopt.name==TWO_POINT_NAMES[2]: # gammat
                gglbins=np.loadtxt(self.input_path("ggl_bins"))
                zbins=np.sum(gglbins[:,2])
                length=np.append(length,zbins*self.params['tbins'])
            elif twopt.name==TWO_POINT_NAMES[3]: # wtheta
                zbins=fits.get_kernel(twopt.kernel1).nbin
                length=np.append(length,zbins*self.params['tbins'])
            else:  #shear-shear xip or xim
                zbins=fits.get_kernel(twopt.kernel1).nbin
                length=np.append(length,(zbins*(zbins+1)/2)*self.params['tbins'])
            if length[-1]!=len(twopt.bin1):
                print('covariance and data vector mismatch in '+name, length[-1], len(twopt.bin1))
                return

        return length.astype(int)


    def sort_2pt(self,fits,length):

        # Sorts 2pt data to match covariance (might duplicate what Joe did in load_twopt_data?)


        for iname,name in enumerate(TWO_POINT_NAMES):
            twopt=fits.get_spectrum(name)
            twopt_order = np.vstack((twopt.angular_bin,twopt.bin1-1,twopt.bin2-1)).T

            mask = []
            for i in range(length[iname]):
                mask.append(np.where((twopt_order[:,0] == self.covorder[np.sum(length[:iname])+i,0])
                    &(twopt_order[:,1] == self.covorder[np.sum(length[:iname])+i,1])
                    &(twopt_order[:,2] == self.covorder[np.sum(length[:iname])+i,2]))[0][0])

            twopt.apply_mask(mask)

        return
