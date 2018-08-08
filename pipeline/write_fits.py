import h5py
import numpy as np
import twopoint
import fitsio as fio
import glob
import yaml
import os
import collections
import blind_2pt_usingcosmosis as blind
from .stage import PipelineStage, TWO_POINT_NAMES, NOFZ_NAMES
import subprocess

print 'Imported modules'

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
        """
        super(WriteFits,self).__init__(param_file)
        
    def run(self):

        # # Initialise twopoint spectrum classes
        # self.init_specs()
        
        # # Load xi data
        # self.load_metadata()
        # self.load_twopt_data()

        # # Load covariance info
        # self.load_cov()

        do_Blinding = True
        if do_Blinding:
            print '\nBlinding will be applied to the file being written\nThis is hard-coded.' 
            print 'To unblind, set do_Blinding=False in write_fits.py'
            self.blind()
            
        return

    def blind(self):
        #Requires sourcing a cosmosis-setup file

        import pickle

        #os.system('source pipeline/BASHTEST.sh > BLINDING_LOG.txt')
        print 'running blinding'
        # command = ['source ~/cosmosis/LOAD_STUFF']
        # subprocess.call(command,shell=True)
        # proc = subprocess.Popen(command,shell=True)   
        # print proc.communicate()     
        print 'running blinding done'

        try:
            source = 'source ~/cosmosis/LOAD_STUFF'
            dump = '/usr/bin/python -c "import os,pickle;print pickle.dumps(os.environ)"'
            penv = os.popen('%s && %s' %(source,dump))
            a=penv.read()
            print a
            env = pickle.loads(a)
            os.environ = env
            try:
                import cosmosis
            except:
                print 'still doesnt work'
        except:

            import os, subprocess as sp, json
            source = 'source init_env'
            dump = '/usr/bin/python -c "import os, json;print json.dumps(dict(os.environ))"'
            pipe = sp.Popen(['/bin/bash', '-c', '%s && %s' %(source,dump)], stdout=sp.PIPE)
            a=pipe.stdout.read()
            env = json.loads(a)
            os.environ = env
            try:
                import cosmosis
            except:
                print 'still doesnt work b'

        #uses Jessie's pipeline to blind the measurement once it's written
        #it basically runs cosmosis twice, once at some fiducial cosmology and then at a randomly-shifted cosmology
        #the blinding factor applied to the measurement is the difference (or ratio) between these 2 cosmologies
        #ini_name = self.self.params['ini']
        #seed_name = self.self.params['seed']
        #btype_name = ' -b '+self.self.params['btype']
        #label_name = ' -t '+self.self.params['label']

        #unblinded_name = self.self.params['run_directory']+'/'+self.name+'/'+self.outputs['2pt_ng']
        #source_command = 'source '+self.self.params['cosmosis_setup']
        #os.system(source_command)
        #blind.do2ptblinding(self.self.params['seed'],self.self.params['ini'],unblinded_name,None,self.self.params['label'],self.self.params['btype'],None)
        

        #unblinded_name =' -u '+self.self.params['run_directory']+'/'+self.name+'/'+self.outputs['2pt_ng']
        #print unblinded_name
        #ini_name = ' -i '+self.self.params['ini']
        #seed_name = ' -s '+self.self.params['seed']
        #btype_name = ' -b '+self.self.params['btype']
        #label_name = ' -t '+self.self.params['label']
        #blinding_command = 'python pipeline/blind_2pt_usingcosmosis.py '+ seed_name + ini_name + btype_name + label_name + unblinded_name
        #print source_command+'\n'+blinding_command
        #os.system(source_command+' \n '+blinding_command)
        
        return

    def strip_wtheta(self, fits):
        if self.params['cross_clustering']:
            return

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

        if self.covmat is not None:
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

        print "Have disabled covmat cleanup"
        # self.cleanup_cov()

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

    def get_pixels(self,f,cf):

        pixels=np.array([],dtype=int)
        for f_ in f:
            pixels=np.append(pixels,np.array(f_['2pt/'+cf].keys(),dtype=int))

        return np.unique(pixels)

    def test_nbin(self,f,cf,i_true,j_true):

        ibins=np.array([],dtype=int)
        jbins=np.array([],dtype=int)
        for f_ in f:
            tomo = f_['2pt/'+cf][f_['2pt/'+cf].keys()[0]]['0']
            ibins=np.append(ibins,np.array(tomo.keys(),dtype=int))
            for key in tomo.keys():
                tomo_ = tomo[key]
                jbins=np.append(jbins,np.array(tomo_.keys(),dtype=int))

        assert np.all(np.unique(ibins)==np.arange(i_true)),'Number of i tomographic bins in '+cf+' different from expectation from stage nofz.'
        assert np.all(np.unique(jbins)==np.arange(j_true)),'Number of j tomographic bins in '+cf+' different from expectation from stage nofz.'

    def load_twopt_data(self):

        # Load links to h5 file output
        f=[]
        for f_ in glob.glob('2pt_*.h5'):
            f.append(h5py.File(f_,mode='r'))

        # Get pixel list from output
        pixels = self.get_pixels(f,'xipm')
        # Test that tomographic bins match expectation
        self.test_nbin(f,'xipm',self.zbins,self.zbins)

        # Do loop to read in xipm
        self.exts[0].bin1        = []
        self.exts[0].bin2        = []
        self.exts[0].angle       = []
        self.exts[0].angular_bin = []
        self.exts[0].value       = []
        self.exts[0].npairs      = []
        self.exts[0].weight      = []
        self.exts[1].bin1        = []
        self.exts[1].bin2        = []
        self.exts[1].angle       = []
        self.exts[1].angular_bin = []
        self.exts[1].value       = []
        self.exts[1].npairs      = []
        self.exts[1].weight      = []
        for t_ in range(self.zbins):
            for t2_ in range(self.zbins):
                if t2_<t_:
                    continue
                self.exts[0].bin1 = np.append(self.exts[0].bin1,np.ones(self.params['tbins'])*int(t_)+1)
                self.exts[1].bin1 = np.append(self.exts[1].bin1,np.ones(self.params['tbins'])*int(t_)+1)
                self.exts[0].bin2 = np.append(self.exts[0].bin2,np.ones(self.params['tbins'])*int(t2_)+1)
                self.exts[1].bin2 = np.append(self.exts[1].bin2,np.ones(self.params['tbins'])*int(t2_)+1)
                weight   = np.zeros(self.params['tbins'])
                npairs   = np.zeros_like(weight)
                meanlogr = np.zeros_like(weight)
                xip      = np.zeros_like(weight)
                xim      = np.zeros_like(weight)
                for p_ in pixels:
                    for p2_ in range(9):
                        for f_ in f:
                            try:
                                xip      += f_['2pt/xipm/'][str(int(p_))][str(int(p2_))][str(int(t_))][str(int(t2_))]['xip'][:]
                                xim      += f_['2pt/xipm/'][str(int(p_))][str(int(p2_))][str(int(t_))][str(int(t2_))]['xim'][:]
                                npairs   += f_['2pt/xipm/'][str(int(p_))][str(int(p2_))][str(int(t_))][str(int(t2_))]['npairs'][:]
                                weight   += f_['2pt/xipm/'][str(int(p_))][str(int(p2_))][str(int(t_))][str(int(t2_))]['weight'][:]
                                meanlogr += f_['2pt/xipm/'][str(int(p_))][str(int(p2_))][str(int(t_))][str(int(t2_))]['meanlogr'][:]
                            except:
                                continue
                self.exts[0].angle       = np.append(self.exts[0].angle,np.exp(meanlogr/weight)*180/np.pi*60)
                self.exts[0].angular_bin = np.append(self.exts[0].angular_bin,np.arange(self.params['tbins'],dtype=int))
                self.exts[0].value       = np.append(self.exts[0].value,xip/weight)
                self.exts[0].npairs      = np.append(self.exts[0].npairs,npairs)
                self.exts[0].weight      = np.append(self.exts[0].weight,weight)
                self.exts[1].angle       = np.append(self.exts[1].angle,np.exp(meanlogr/weight)*180/np.pi*60)
                self.exts[1].angular_bin = np.append(self.exts[1].angular_bin,np.arange(self.params['tbins'],dtype=int))
                self.exts[1].value       = np.append(self.exts[1].value,xim/weight)
                self.exts[1].npairs      = np.append(self.exts[1].npairs,npairs)
                self.exts[1].weight      = np.append(self.exts[1].weight,weight)

        # Get pixel list from output
        pixels = self.get_pixels(f,'gammat')
        # Test that tomographic bins match expectation
        self.test_nbin(f,'gammat',self.lens_zbins,self.zbins)

        # Do loop to read in gammat/gammax
        self.exts[2].bin1        = []
        self.exts[2].bin2        = []
        self.exts[2].angle       = []
        self.exts[2].angular_bin = []
        self.exts[2].value       = []
        self.exts[2].npairs      = []
        self.exts[2].weight      = []
        self.exts[2].random_npairs = []
        self.exts[2].random_weight = []
        self.exts[3].bin1        = []
        self.exts[3].bin2        = []
        self.exts[3].angle       = []
        self.exts[3].angular_bin = []
        self.exts[3].value       = []
        self.exts[3].npairs      = []
        self.exts[3].weight      = []
        self.exts[3].random_npairs = []
        self.exts[3].random_weight = []
        for t_ in range(self.lens_zbins):
            for t2_ in range(self.zbins):
                self.exts[2].bin1 = np.append(self.exts[2].bin1,np.ones(self.params['tbins'])*int(t_)+1)
                self.exts[3].bin1 = np.append(self.exts[3].bin1,np.ones(self.params['tbins'])*int(t_)+1)
                self.exts[2].bin2 = np.append(self.exts[2].bin2,np.ones(self.params['tbins'])*int(t2_)+1)
                self.exts[3].bin2 = np.append(self.exts[3].bin2,np.ones(self.params['tbins'])*int(t2_)+1)
                ngweight  = np.zeros(self.params['tbins'])
                ngnpairs  = np.zeros_like(ngweight)
                rgweight  = np.zeros_like(ngweight)
                rgnpairs  = np.zeros_like(ngweight)
                meanlogr  = np.zeros_like(ngweight)
                ngxi      = np.zeros_like(ngweight)
                ngxim     = np.zeros_like(ngweight)
                rgxi      = np.zeros_like(ngweight)
                rgxim     = np.zeros_like(ngweight)
                for p_ in pixels:
                    for p2_ in range(9):
                        for f_ in f:
                            try:
                                ngxi       += f_['2pt/gammat/'][str(int(p_))][str(int(p2_))][str(int(t_))][str(int(t2_))]['ngxi'][:]
                                ngxim      += f_['2pt/gammat/'][str(int(p_))][str(int(p2_))][str(int(t_))][str(int(t2_))]['ngxim'][:]
                                ngnpairs   += f_['2pt/gammat/'][str(int(p_))][str(int(p2_))][str(int(t_))][str(int(t2_))]['ngnpairs'][:]
                                ngweight   += f_['2pt/gammat/'][str(int(p_))][str(int(p2_))][str(int(t_))][str(int(t2_))]['ngweight'][:]
                                rgxi       += f_['2pt/gammat/'][str(int(p_))][str(int(p2_))][str(int(t_))][str(int(t2_))]['rgxi'][:]
                                rgxim      += f_['2pt/gammat/'][str(int(p_))][str(int(p2_))][str(int(t_))][str(int(t2_))]['rgxim'][:]
                                rgnpairs   += f_['2pt/gammat/'][str(int(p_))][str(int(p2_))][str(int(t_))][str(int(t2_))]['rgnpairs'][:]
                                rgweight   += f_['2pt/gammat/'][str(int(p_))][str(int(p2_))][str(int(t_))][str(int(t2_))]['rgweight'][:]
                                meanlogr   += f_['2pt/gammat/'][str(int(p_))][str(int(p2_))][str(int(t_))][str(int(t2_))]['meanlogr'][:]
                            except:
                                continue
                self.exts[2].angle         = np.append(self.exts[2].angle,np.exp(meanlogr/ngweight)*180/np.pi*60)
                self.exts[2].angular_bin   = np.append(self.exts[2].angular_bin,np.arange(self.params['tbins'],dtype=int))
                self.exts[2].value         = np.append(self.exts[2].value,ngxi/ngweight-rgxi/rgweight)
                self.exts[2].npairs        = np.append(self.exts[2].npairs,ngnpairs)
                self.exts[2].weight        = np.append(self.exts[2].weight,ngweight)
                self.exts[2].random_npairs = np.append(self.exts[2].random_npairs,rgnpairs)
                self.exts[2].random_weight = np.append(self.exts[2].random_weight,rgweight)
                self.exts[3].angle         = np.append(self.exts[3].angle,np.exp(meanlogr/ngweight)*180/np.pi*60)
                self.exts[3].angular_bin   = np.append(self.exts[3].angular_bin,np.arange(self.params['tbins'],dtype=int))
                self.exts[3].value         = np.append(self.exts[3].value,ngxim/ngweight-rgxim/rgweight)
                self.exts[3].npairs        = np.append(self.exts[3].npairs,ngnpairs)
                self.exts[3].weight        = np.append(self.exts[3].weight,ngweight)
                self.exts[3].random_npairs = np.append(self.exts[3].random_npairs,rgnpairs)
                self.exts[3].random_weight = np.append(self.exts[3].random_weight,rgweight)

        # Get pixel list from output
        pixels = self.get_pixels(f,'wtheta')
        # Test that tomographic bins match expectation
        self.test_nbin(f,'wtheta',self.lens_zbins,self.lens_zbins)

        # Do loop to read in wtheta
        self.exts[4].bin1          = []
        self.exts[4].bin2          = []
        self.exts[4].angle         = []
        self.exts[4].angular_bin   = []
        self.exts[4].value         = []
        self.exts[4].npairs        = []
        self.exts[4].weight        = []
        self.exts[4].random_npairs = []
        self.exts[4].random_weight = []
        for t_ in range(self.lens_zbins):
            for t2_ in range(self.lens_zbins):
                if t_>t2_:
                    continue
                self.exts[4].bin1 = np.append(self.exts[4].bin1,np.ones(self.params['tbins'])*int(t_)+1)
                self.exts[4].bin2 = np.append(self.exts[4].bin2,np.ones(self.params['tbins'])*int(t2_)+1)
                nnweight  = np.zeros(self.params['tbins'])
                nnnpairs  = np.zeros_like(nnweight)
                nrweight  = np.zeros_like(nnweight)
                nrnpairs  = np.zeros_like(nnweight)
                rnweight  = np.zeros_like(nnweight)
                rnnpairs  = np.zeros_like(nnweight)
                rrweight  = np.zeros_like(nnweight)
                rrnpairs  = np.zeros_like(nnweight)
                meanlogr  = np.zeros_like(nnweight)
                nntot=0
                rntot=0
                nrtot=0
                rrtot=0
                for p_ in pixels:
                    for p2_ in range(9):
                        for f_ in f:
                            try:
                                if len(f_['2pt/wtheta/'][str(int(p_))][str(int(p2_))][str(int(t_))][str(int(t2_))]['nntot'][:])==1:
                                    if f_['2pt/wtheta/'][str(int(p_))][str(int(p2_))][str(int(t_))][str(int(t2_))]['nntot'][:] ==0:
                                        continue
                                    if f_['2pt/wtheta/'][str(int(p_))][str(int(p2_))][str(int(t_))][str(int(t2_))]['nrtot'][:] ==0:
                                        continue
                                    if f_['2pt/wtheta/'][str(int(p_))][str(int(p2_))][str(int(t_))][str(int(t2_))]['rntot'][:] ==0:
                                        continue
                                    if f_['2pt/wtheta/'][str(int(p_))][str(int(p2_))][str(int(t_))][str(int(t2_))]['rrtot'][:] ==0:
                                        continue
                                nnnpairs   += f_['2pt/wtheta/'][str(int(p_))][str(int(p2_))][str(int(t_))][str(int(t2_))]['nnnpairs'][:]
                                nnweight   += f_['2pt/wtheta/'][str(int(p_))][str(int(p2_))][str(int(t_))][str(int(t2_))]['nnweight'][:]
                                nrnpairs   += f_['2pt/wtheta/'][str(int(p_))][str(int(p2_))][str(int(t_))][str(int(t2_))]['nrnpairs'][:]
                                nrweight   += f_['2pt/wtheta/'][str(int(p_))][str(int(p2_))][str(int(t_))][str(int(t2_))]['nrweight'][:]
                                rnnpairs   += f_['2pt/wtheta/'][str(int(p_))][str(int(p2_))][str(int(t_))][str(int(t2_))]['rnnpairs'][:]
                                rnweight   += f_['2pt/wtheta/'][str(int(p_))][str(int(p2_))][str(int(t_))][str(int(t2_))]['rnweight'][:]
                                rrnpairs   += f_['2pt/wtheta/'][str(int(p_))][str(int(p2_))][str(int(t_))][str(int(t2_))]['rrnpairs'][:]
                                rrweight   += f_['2pt/wtheta/'][str(int(p_))][str(int(p2_))][str(int(t_))][str(int(t2_))]['rrweight'][:]
                                meanlogr   += f_['2pt/wtheta/'][str(int(p_))][str(int(p2_))][str(int(t_))][str(int(t2_))]['meanlogr'][:]
                                nntot      += f_['2pt/wtheta/'][str(int(p_))][str(int(p2_))][str(int(t_))][str(int(t2_))]['nntot'][:]
                                nrtot      += f_['2pt/wtheta/'][str(int(p_))][str(int(p2_))][str(int(t_))][str(int(t2_))]['nrtot'][:][0]
                                rntot      += f_['2pt/wtheta/'][str(int(p_))][str(int(p2_))][str(int(t_))][str(int(t2_))]['rntot'][:][0]
                                rrtot      += f_['2pt/wtheta/'][str(int(p_))][str(int(p2_))][str(int(t_))][str(int(t2_))]['rrtot'][:][0]
                            except:
                                continue
                self.exts[4].angle         = np.append(self.exts[4].angle,np.exp(meanlogr/nnweight)*180/np.pi*60)
                self.exts[4].angular_bin   = np.append(self.exts[4].angular_bin,np.arange(self.params['tbins'],dtype=int))
                rrw = 1.*nntot / rrtot
                drw = 1.*nntot / rntot
                rdw = 1.*nntot / rntot
                xi = (nnweight - rnweight * rdw - nrweight * drw + rrweight * rrw) / (rrweight * rrw) 
                self.exts[4].value         = np.append(self.exts[4].value,xi)
                self.exts[4].npairs        = np.append(self.exts[4].npairs,nnnpairs)
                self.exts[4].weight        = np.append(self.exts[4].weight,nnweight)
                self.exts[4].random_npairs = np.append(self.exts[4].random_npairs,rrnpairs)
                self.exts[4].random_weight = np.append(self.exts[4].random_weight,rrweight)

        for f_ in f:
            f_.close()

    def load_cov(self):
        import os

        #if os.path.exists(self.input_path("cov")): 
        #    os.remove(self.input_path("cov"))
        #os.system("cat "+self.input_path("covfiles")+" >> "+self.input_path("cov"))
        try:
            covdata = np.loadtxt(self.input_path("cov"))
        except:
            self.covmat = None
            print 'Skipping covariance, since output file missing.'
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
                print 'covariance and data vector mismatch in '+name, length[-1], len(twopt.bin1)
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

