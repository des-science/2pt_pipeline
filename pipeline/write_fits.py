import numpy as np
import twopoint
import fitsio as fio
from .stage import PipelineStage, TWO_POINT_NAMES, NOFZ_NAMES
import glob
import collections
import os

class WriteFits(PipelineStage):
    name = "2pt_fits"
    inputs = {
        "ggl_bins"              : ("cov", "run.gglensing_zbins"),
        "cov"                   : ("cov", "cov.txt"),
        "covfiles"              : ("cov", "run_*"),

        "2pt"                   : ("nofz","2pt.fits"),
        "xip"                   : ("2pt", "*_xip.txt"),
        "xim"                   : ("2pt", "*_xim.txt"),
        "gammat"                : ("2pt", "*_gammat.txt"),
        "wtheta"                : ("2pt", "*_wtheta.txt"),
    }
    outputs = {
        "2pt_extended"          : "2pt_extended_data.fits",
        "2pt_ng"                : "2pt_NG.fits",
        "2pt_g"                 : "2pt_G.fits",
    }

    def __init__(self,param_file):
        """
        Initialise object.
        """
        super(WriteFits,self).__init__(param_file)

    def run(self):

        # Initialise twopoint spectrum classes
        #self.init_specs()
        
        # Load xi data
        #self.load_twopt_data()

        # Load covariance info
        #self.load_cov()

        do_Blinding = True
        if do_Blinding:
            print '\nBlinding will be applied to the file being written\nThis is hard-coded.' 
            print 'To unblind, set do_Blinding=False in write_fits.py'
            self.blind()
        
        return

    def blind(self):
        #Requires sourcing a cosmosis-setup file

        #uses Jessie's pipeline to blind the measurement once it's written
        #it basically runs cosmosis twice, once at some fiducial cosmology and then at a randomly-shifted cosmology
        #the blinding factor applied to the measurement is the difference (or ratio) between these 2 cosmologies
        print 'BLINDING GOES HERE! '
        source_command = 'source '+self.params['cosmosis_setup']
        os.system(source_command) #change to a more general location
        unblinded_name = self.params['run_directory']+'/'+self.name+'/'+self.outputs['2pt_ng']
        print unblinded_name
        run_cosmosis_command = os.system()

        return

    def strip_wtheta(self, fits):
#        if self.params['cross_clustering']:
#            return

        wtheta = fits.get_spectrum(TWO_POINT_NAMES[3])
        mask = wtheta.bin1==wtheta.bin2
        print "Cutting out {} values from wtheta because no cross_clustering".format(len(mask)-mask.sum())
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

        print "Cutting out {} values from gammat because lens behind source".format(len(mask)-mask.sum())

        gammat.apply_mask(mask)


    def write(self):

        # Load 2point fits file with n(z) info.
        fits=twopoint.TwoPointFile.from_fits(self.input_path("2pt"),covmat_name=None)

        # Write file without covariance (all data vectors)
        fits.spectra=self.exts
        fits.to_fits(self.output_path("2pt_extended"), clobber=True)

        if self.params['lensfile'] != 'None':
            self.strip_wtheta(fits)
            self.strip_missing_gglensing(fits)

        length=self.get_cov_lengths(fits)

        self.sort_2pt(fits,length)

        # Writes the covariance info into a covariance object and saves to 2point fits file. 
        fits.covmat_info=twopoint.CovarianceMatrixInfo('COVMAT',TWO_POINT_NAMES,length,self.covmat[0])
        fits.to_fits(self.output_path("2pt_g"),clobber=True)
        fits.covmat_info=twopoint.CovarianceMatrixInfo('COVMAT',TWO_POINT_NAMES,length,self.covmat[1])
        fits.to_fits(self.output_path("2pt_ng"),clobber=True)

        print "Have disabled covmat cleanup"
#        self.cleanup_cov()


    def load_2pt_text_fmt(self, name):
        data = []
        filenames = glob.glob(self.input_path(name))
        for filename in filenames:
            d = np.loadtxt(filename)
            data.append(d)
        if len(data)>1:
            data = np.vstack(data)
        else:
            data = data[0]
        theta = data[:,0]
        ibin = data[:,1].astype(int)
        jbin = data[:,2].astype(int)
        value = data[:,3]
        npairs = data[:,4]
        return theta, ibin, jbin, value,npairs

    def load_twopt_data(self):

        if self.params['lensfile'] != 'None':
            names = TWO_POINT_NAMES
        else:
            names = TWO_POINT_NAMES[:2]

        for i,name in enumerate(names):
            theta, ibin, jbin, value, npairs =  self.load_2pt_text_fmt(name)

            mask=np.lexsort((jbin,ibin))
            ibin = ibin[mask]
            jbin = jbin[mask]
            theta = theta[mask]
            value = value[mask]

            # make angular bins, assuming that
            # they are grouped by pairs. really 
            # hope that is the case. tried to write a test
            # but brain fried.  sorry future person this screws over.
            pairs = collections.OrderedDict()
            for i1,j1 in zip(ibin,jbin):
                if (i1,j1) not in pairs:
                    pairs[(i1,j1)] = 0
                pairs[(i1,j1)] += 1
            npair = len(pairs)
            angbins = []
            for pair,count in pairs.items():
                angbins.append(np.arange(count))
            angbins = np.concatenate(angbins)

            #tomo measurements
            self.exts[i].bin1=ibin+1
            self.exts[i].bin2=jbin+1
            self.exts[i].angle=theta
            self.exts[i].angular_bin=angbins
            self.exts[i].value=value
            self.exts[i].npairs=npairs

    def load_cov(self):
        import os

        #if os.path.exists(self.input_path("cov")): 
        #    os.remove(self.input_path("cov"))
        #os.system("cat "+self.input_path("covfiles")+" >> "+self.input_path("cov"))
        covdata = np.loadtxt(self.input_path("cov"))

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

        dtype1=[twopoint.Types.galaxy_shear_plus_real,twopoint.Types.galaxy_shear_minus_real,twopoint.Types.galaxy_position_real,twopoint.Types.galaxy_position_real]
        dtype2=[twopoint.Types.galaxy_shear_plus_real,twopoint.Types.galaxy_shear_minus_real,twopoint.Types.galaxy_shear_plus_real,twopoint.Types.galaxy_position_real]
        nznameindex1=[0,0,1,1]
        nznameindex2=[0,0,0,1]

        
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

        if self.params['lensfile'] == 'None':
            self.exts=self.exts[:2]


        return 

    def get_cov_lengths(self,fits):

        # Calculate length of covariance blocks. Exception for gammat, which reads a file that stores which bin combinations have been rejected in the covariance calcultion.
        if self.params['lensfile'] != 'None':
            names = TWO_POINT_NAMES
        else:
            names = TWO_POINT_NAMES[:2]

        # Make cov lengths
        length=np.array([])
        for name in names:
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
                print 'covariance and data vector mismatch in '+name, length[-1], len(twopt.bin1)
                return

        return length.astype(int)


    def sort_2pt(self,fits,length):

        # Sorts 2pt data to match covariance (might duplicate what Joe did in load_twopt_data?)

        if self.params['lensfile'] != 'None':
            names = TWO_POINT_NAMES
        else:
            names = TWO_POINT_NAMES[:2]

        for iname,name in enumerate(names):
            twopt=fits.get_spectrum(name)
            twopt_order = np.vstack((twopt.angular_bin,twopt.bin1-1,twopt.bin2-1)).T

            mask = []
            for i in range(length[iname]):
                mask.append(np.where((twopt_order[:,0] == self.covorder[np.sum(length[:iname])+i,0])
                    &(twopt_order[:,1] == self.covorder[np.sum(length[:iname])+i,1])
                    &(twopt_order[:,2] == self.covorder[np.sum(length[:iname])+i,2]))[0][0])

            twopt.apply_mask(mask)

        return

