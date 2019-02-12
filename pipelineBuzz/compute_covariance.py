from .stage import PipelineStage
import twopoint
import numpy as np
import glob
import os
import sys

def run_cmd(cmd):
    print "Running command:"
    print cmd
    print
    os.system(cmd)


class ComputeCovariance(PipelineStage):
    """
    A class for interfacing with the cosmolike halo model covariance c code. It produces an ini file for the covariance calculation based on what n(z) and 2pt data vectors are present in the 2point fits file pointed to. This works now for xi+, xi-, gammat, wtheta, allowing separate lens and source n(z)s. It calls/outputs call for the covariance calculation and then imports the resulting output files into the 2point fits file for use in cosmosis/cosmolike for parameter estimation.

    Example syntax for use is shown in __main__.
    """
    name = "cov"
    inputs = {
        "nz_source_txt" : ("nofz", "source.nz"),
        "nz_lens_txt"   : ("nofz", "lens.nz"),
        "metadata"      : ("nofz", "metadata.yaml"),

    }
    outputs = {
        # loads of covariance files - not listed separately.
        "cov_ini"     : "cov.ini"            ,
        "cov"         : "cov.txt",
        "cov_chunks"  : "run*cov_Ntheta*",
        "blocks"      : "run.blocks",
        "ggl_bins"    : "run.gglensing_zbins",
    }


    def __init__(self, param_file):
        """
        Read in properties of data vectors and n(z)s from 2point fits file provided in twopt_pipeline.ini file.
        """
        super(ComputeCovariance,self).__init__(param_file)

    def run(self):
        if self.comm is None:
            print "Running single core"
            self.run_single_core()
        else:
            print "Running MPI"
            self.run_mpi()

    def run_mpi(self):
        from .mpi_pool import MPIPool
        pool = MPIPool(comm=self.comm,debug=True)
        
        if pool.is_master():
            self.prepare_covariance_runs()
            commands = self.generate_commands()
            print "will run these commands:"
            for command in commands:
                print command
            sys.stdout.flush()
            print
            print
        else:
            commands = None
        self.comm.Barrier()
        pool.map(run_cmd, commands)
        pool.close()
        self.comm.Barrier()


    def run_single_core(self):
        self.prepare_covariance_runs()
        for command in self.generate_commands():
            print command
            status = os.system(command)
            if status!=0:
                raise ValueError("Failed command: {}".format(command))


    def generate_commands(self):
        """
        Uses loaded in info from __init__ to write n(z) and ini files for cosmolike covariance. Calls/produces call to run that code (still in progress).
        """

        # Import number of parallel blocks.
        filename = self.output_path("blocks")
        blocknum=np.loadtxt(filename).astype(int)

        commands=[]
        # Build list of os system calls
        ini = self.output_path("cov_ini")
        exe = self.params['cov_source_dir']
        for i in range(blocknum):
            cmd = "{} {} {}".format(exe, ini, i+1)
            commands.append(cmd)

        return commands

    def write(self):
        """
        Empty - writing is done at c layer.
        """

        if not (self.comm is None or self.comm.Get_rank()==0):
            return

        return

        import glob
        files = glob.glob(self.output_path("cov_chunks"))

        print "final number of files = ", len(files)
        #int int double double int int int int double double
        cov_filename = self.output_path("cov")
        if os.path.exists(cov_filename):
            os.remove(cov_filename)

        for filename in files:
            os.system("cat {} >> {}".format(filename, cov_filename))
        
#        self.cleanup_cov()


    def cleanup_cov(self):
        """
        Cleanup covariance output files once stored in collated text file.
        """

        import os
        for ifile,file in enumerate(glob.glob(self.params['outdir']+self.params['prefix']+'_*_cov_*')):
            os.remove(file)

        return


    def prepare_covariance_runs(self):
        """
        Write a covariance ini file, based on a default dictionary of values. Uses imported binning and file pointers info from 2point fits file and pipeline.ini file. 
        """

        self.load_metadata()
        
        # Default dictionary.
        cov_dict = {
        'Omega_m'                   : 0.286,
        'Omega_v'                   : 0.714,
        'sigma_8'                   : 0.82,
        'n_spec'                    : 0.96,
        'w0'                        : -1.0,
        'wa'                        : 0.0,
        'omb'                       : 0.05,
        'h0'                        : 0.7,
        'coverH0'                   : 2997.92458,
        'rho_crit'                  : 7.4775e+21,
        'f_NL'                      : 0.0,
        'pdelta_runmode'            : 'Halofit',
        'area'                      : self.area,
        'source_n_gal'              : self.neff,
        'source_tomobins'           : self.tomobins,
        'sigma_e'                   : np.mean(self.sigma_e)*np.sqrt(2.),
        'm_lim'                     : 24.0,
        'Kcorrect_File'             : '../zdistris/k+e.dat',
        'shear_REDSHIFT_FILE'       : self.input_path("nz_source_txt"),
        'sourcephotoz'              : 'multihisto',
        'lensphotoz'                : 'multihisto',
        'tmin'                      : self.params['tbounds'][0],
        'tmax'                      : self.params['tbounds'][1],
        'ntheta'                    : self.params['tbins'],
        'outdir'                    : self.output_dir+"/",
        'filename'                  : "run",
        'ggl_overlap_cut'           : self.params['ggl_overlap_cut'],
        'ss'                        : 'true',
        'ng'                        : 0
        }

        if self.params['lensfile'] != 'None':
            cov_dict.update({'clustering_REDSHIFT_FILE' : self.input_path("nz_lens_txt"),
                            'lens_tomobins'             : self.lens_tomobins,
                            'lens_n_gal'                : self.lens_neff,
                            'ls'                        : 'true',
                            'll'                        : 'true',
                            'lens_tomogbias'            : self.params['lens_gbias']})
        else:
            cov_dict.update({'clustering_REDSHIFT_FILE' : self.input_path("nz_source_txt"),
                            'lens_tomobins'             : self.tomobins,
                            'lens_n_gal'                : self.neff,
                            'ls'                        : 'false',
                            'll'                        : 'false',
                            'lens_tomogbias'            : np.ones(self.tomobins)})

        # Write ini file.
        filename = self.output_path("cov_ini")

        if self.params['ng']:
            cov_dict['ng']=1

        with open(filename,'w') as f:
            f.write('#\n')
            f.write('# Cosmological parameters\n')
            f.write('#\n')

            for x in ['Omega_m', 'Omega_v', 'sigma_8', 'n_spec', 'w0', 'wa', 'omb', 'h0', 'coverH0', 'rho_crit', 'f_NL', 'pdelta_runmode']:
                f.write(x + ' : ' + str(cov_dict[x]) + '\n')

            f.write('#\n')
            f.write('# Survey and galaxy parameters\n')
            f.write('#\n')
            f.write('# area in degrees\n')
            f.write('# n_gal,lens_n_gal in gals/arcmin^2\n')

            for x in ['area', 'sourcephotoz', 'lensphotoz', 'source_tomobins', 'lens_tomobins', 'sigma_e', 'shear_REDSHIFT_FILE', 'clustering_REDSHIFT_FILE']:
                f.write(x + ' : ' + str(cov_dict[x]) + '\n')                    

            for x in ['source_n_gal', 'lens_n_gal', 'lens_tomogbias']:
                if hasattr(cov_dict[x],"__len__"):
                    f.write(x + ' : ' + ','.join(np.around(cov_dict[x],4).astype(str)) + '\n')
                else:
                    f.write(x + ' : ' + str(cov_dict[x]) + '\n')                    

            f.write('#\n')
            f.write('# Covariance paramters\n')
            f.write('#\n')
            f.write('# tmin,tmax in arcminutes\n')
            for x in ['tmin', 'tmax', 'ntheta', 'outdir', 'filename', 'ss', 'ls', 'll','ggl_overlap_cut','ng']:
                f.write(x + ' : ' + str(cov_dict[x]) + '\n')

        # Remove old files. Call zero block covariance to write out number of parallel blocks and excluded gammat bin pairs.
        import glob
        files = glob.glob(self.output_path("cov_chunks"))
        for file in files:
            os.remove(file)
        exe = self.params['cov_source_dir']
        ini = self.output_path("cov_ini")
        command = "{} {} 0".format(exe, ini, 0)
        print "Running command:", command
        os.system(command)

    def load_metadata(self):
        import yaml
        filename = self.input_path('metadata')
        data = yaml.load(open(filename))
        self.neff = np.array(data['neff'])
        self.tomobins = data['tomobins']
        self.sigma_e = np.array(data['sigma_e'])
        self.area = data['area']
        if self.params['lensfile'] != 'None':
            self.lens_neff = np.array(data['lens_neff'])
            self.lens_tomobins = data['lens_tomobins']

