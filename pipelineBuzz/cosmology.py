from .stage import PipelineStage
import os
import yaml
import copy
import collections
import string
import numpy as np

class SafeFormatter(string.Formatter):
    """
    This helper class from http://stackoverflow.com/questions/17215400/python-format-string-unused-named-arguments
    means that we can keep the {} placeholders in the cosmosis parameter file that are looking
    for environment variables and use this to format it to replace other parameters.
    """
    def __init__(self, default='{{{0}}}'):
        self.default=default

    def get_value(self, key, args, kwds):
        if isinstance(key, str):
            return kwds.get(key, self.default.format(key))
        else:
            Formatter.get_value(key, args, kwds)

formatter = SafeFormatter()



this_dir_name = os.path.abspath(os.path.split(__file__)[0])

class ParameterEstimation(PipelineStage):
    name = "cosmo"
    inputs = {
        "2pt_text"          : ("2pt_text", "2pt.txt"),
        "mask"              : ("2pt_text", "mask.txt" ),
        "nz_source_text"    : ("2pt_text", "source_nz.txt"),
        "nz_lens_text"      : ("2pt_text", "lens_nz.txt"),
        "cov_txt"           : ("cov",      "cov.txt"  ),
        "2pt_fits"          : ("2pt_fits", "2pt_NG.fits"),
        "2pt_fits_gaussian" : ("2pt_fits", "2pt_G.fits"),
    }
    outputs = {
        "cosmolike_fid_datavector" : "cosmolike_fid_datavector.txt",
        "cosmosis_params" : "cosmosis_params.ini",
        "cosmosis_test_params" : "cosmosis_test_params.ini",
        "cosmosis_values" : "cosmosis_values.ini",
        "cosmosis_priors" : "cosmosis_priors.ini",
        "cosmosis_chain"  : "chain.txt",
        "cosmosis_theory_data"  : "cosmosis_theory_data",
        "cosmosis_2pt_fits"  : "cosmosis_2pt.fits",
    }
    def __init__(self, param_file):
        super(ParameterEstimation,self).__init__(param_file)

    def run(self):
        code = self.params['cosmo_code'].lower()
        if code=='cosmosis':
            self.run_cosmosis()
        elif code=='cosmolike':
            self.run_cosmolike()
        else:
            raise ValueError("Unknown code {}".format(code))

    def flat_prior_to_range(self, param):
        param_name = param+"_range"
        X = self.params[param_name]
        if np.isscalar(X):
            x = X
        elif len(X)==1:
            x = X[0]
        else:
            if len(X)!=3:
                raise ValueError("Cannot understand parameter setting {}:  {}".format(param_name, X))
            x = "  ".join(str(a) for a in X)
        return param_name, str(x)

    def gaussian_prior_to_range(self, param, cosmosis_param):
        param_name = param+"_range"
        mu = np.array(self.params[param+'_mean'])
        sigma_list = self.params.get(param+'_sigma')
        # Fixed parameter value if either the _sigma is not found
        # or sigma==0 (for each parameter separately)
        if sigma_list is None:
            ranges = [str(mu_i) for mu_i in mu]
        else:
            sigma = np.array(sigma_list)
            mins = mu-4*sigma
            maxs = mu+4*sigma
            ranges = []
            for min_i, mu_i, max_i in zip(mins, mu, maxs):
                if min_i==max_i:
                    #i.e. sigma==0; make this parameter fixed
                    range_i = str(mu_i)
                else:
                    #parameter varied: use range.
                    range_i = "{}  {}  {}".format(min_i, mu_i, max_i)
                ranges.append(range_i)
        output=""
        for i,r  in enumerate(ranges):
            output+="{}{} = {}\n".format(cosmosis_param, i+1, r)
        return param_name, output

    def gaussian_priors_to_cosmosis_priors(self, param, cosmosis_param):
        param_name = param + "_priors"
        mu = np.array(self.params[param+'_mean'])
        sigma_list = self.params.get(param+'_sigma')
        # Fixed parameter value if either the _sigma is not found
        # or sigma==0 (for each parameter separately)
        if sigma_list is None:
            return param_name, ""
        output=""
        for i,(mu_i,sigma_i)  in enumerate(zip(mu,sigma_list)):
            name = "{}{}".format(cosmosis_param, i+1)
            output+="{} = gaussian {} {}\n".format(name, mu_i, sigma_i)
        return param_name, output

    def make_cosmosis_ini(self):
        self.make_cosmosis_values()
        self.make_cosmosis_priors()
        self.make_cosmosis_params()


    def make_cosmosis_params(self):
        params_template = open(os.path.join(this_dir_name, 'cosmosis_params_template.ini')).read()
        modified_params = {}
        modified_params['priors_path'] = self.output_path('cosmosis_priors')
        modified_params['values_path'] = self.output_path('cosmosis_values')
        modified_params['chain_path'] = self.output_path('cosmosis_chain')
        modified_params['theory_data_path'] = self.output_path('cosmosis_theory_data')

        if self.params['ng']:
            modified_params['2pt_fits'] = self.input_path("2pt_fits")
        else:
            modified_params['2pt_fits'] = self.input_path("2pt_fits_gaussian")

        twoptnames = self.params['twoptnames']
        do_xip = "xip" in twoptnames
        do_xim = "xim" in twoptnames
        do_xi  = do_xip or do_xim
        do_gammat = "gammat" in twoptnames
        do_wtheta = "wtheta" in twoptnames
        do_source = do_xi or do_gammat
        do_lens = do_wtheta or go_gammat

        likelihood = ""
        if do_xip:
            likelihood += "xip "
        if do_xim:
            likelihood += "xim "
        if do_gammat:
            likelihood += "gammat "
        if do_wtheta:
            likelihood += "wtheta "

        modified_params["no_bias"]      = "no_bias"      if do_lens   else ""
        modified_params["binwise_bias"] = "binwise_bias" if do_lens   else ""
        modified_params["shear_m_bias"] = "shear_m_bias" if do_source else ""
        modified_params["2pt_xi"]       = "2pt_xi"       if do_xi     else ""
        modified_params["2pt_gammat"]   = "2pt_gammat"   if do_gammat else ""
        modified_params["2pt_wtheta"]   = "2pt_wtheta"   if do_wtheta else ""
        modified_params["source_nz"]    = "source"       if do_source else ""
        modified_params["lens_nz"]      = "lens"         if do_lens else ""
        modified_params["photoz_bias_source"]    = "photoz_bias_source"       if do_source else ""
        modified_params["photoz_bias_lens"]      = "photoz_bias_lens"         if do_lens else ""

        #We do it twice once saving the test params with this and once for the mcmc without
        modified_params["save_2pt"]     = ""
        modified_params["sampler"]      = "multinest"


        modified_params["theta_bounds"]      = self.params['tbounds']
        modified_params["n_theta_bins"]      = self.params['tbins']
        modified_params["cosmosis_2pt_fits"] = self.output_path('cosmosis_2pt_fits')

        modified_params["shear_shear"]       = "source-source" if do_xi     else "F"
        modified_params["position_shear"]    = "lens-source"   if do_gammat else "F"
        modified_params["position_position"] = "lens-lens"     if do_wtheta else "F"

        params = formatter.format(params_template, **modified_params)

        #Now the test params
        modified_params["save_2pt"]     = "save_2pt"
        modified_params["sampler"]      = "test"
        test_params = formatter.format(params_template, **modified_params)

        #Write the parameters to ini files.
        filename = self.output_path('cosmosis_params')
        test_filename = self.output_path('cosmosis_test_params')

        f = open(filename,'w')
        f.write(params)
        f.close()

        f = open(test_filename,'w')
        f.write(test_params)
        f.close()


    def make_cosmosis_priors(self):
        priors_template = open(os.path.join(this_dir_name, 'cosmosis_priors_template.ini')).read()
        modified_params = self.params.copy()
        for param, cosmosis_param in [("source_z_bias", "bias_"), ("lens_z_bias","bias_"), ("shear_m", "m")]:
            name, value = self.gaussian_priors_to_cosmosis_priors(param, cosmosis_param)
            modified_params[name] = value
        print modified_params.keys()
        priors = priors_template.format(**modified_params)
        open(self.output_path('cosmosis_priors'),'w').write(priors)

    def make_cosmosis_values(self):
        values_template = open(os.path.join(this_dir_name, 'cosmosis_values_template.ini')).read()
        modified_params=self.params.copy()
        for param in ["omega_m","sigma_8","n_s","w0","wa","omega_b","h0","bias","bias2","A_z"]:
            name, value = self.flat_prior_to_range(param)
            modified_params[name] = value

        # Handle bias separately - cosmolike wants a single range for all the bins
        # turn that into the multiple values that cosmosis wants
        nbin_lens = len(self.params['lens_z_bias_mean'])
        _,bias_range = self.flat_prior_to_range('bias')
        modified_params['bias_range'] = '\n'.join(['b_{} = {}'.format(i+1, bias_range) for i in xrange(nbin_lens)])


        for param, cosmosis_param in [("source_z_bias", "bias_"), ("lens_z_bias","bias_"), ("shear_m", "m")]:
            name, value = self.gaussian_prior_to_range(param, cosmosis_param)
            modified_params[name] = value

        values = values_template.format(**modified_params)
        open(self.output_path('cosmosis_values'),'w').write(values)

    def write(self):
        pass #writing delegated to cosmo codes for now.

    def run_cosmosis(self):
        import imp
        cosmosis_dir = os.environ['COSMOSIS_SRC_DIR']
        cosmosis_exe_path = os.path.join(cosmosis_dir, "bin/cosmosis")
        cosmosis_exe = imp.load_source('cosmosis_exe', cosmosis_exe_path)


        # The main cosmosis function takes some argparse output
        # but all it really does with it is look at some specific
        # attributes.  We can just mock those up here.
        class Args(object):
            pass
        args = Args()

        args.variables = []
        args.params = []


        inifile = self.output_path('cosmosis_params')
        test_inifile = self.output_path('cosmosis_test_params')

        # run once with a single core with the "test" sampler
        # to generate an output data vector
        if self.comm is None or self.comm.Get_rank()==0:
            self.make_cosmosis_ini()
            args.inifile = test_inifile
            args.mpi = False
            cosmosis_exe.main(args)

        if self.comm is not None:
            print "<Waiting for master to finish test run>"
            self.comm.Barrier()
        # reset to using the original sampler again and run
        # MPI Pool is handled within cosmosis
        args.inifile = inifile
        args.mpi = (self.comm is not None)
        
        if args.mpi:
            with cosmosis_exe.mpi_pool.MPIPool() as pool:
                cosmosis_exe.main(args,pool)
        else:
            cosmosis_exe.main(args)


    def run_cosmolike(self):
        if self.comm:
            from .mpi_pool import MPIPool
            pool = MPIPool(self.comm)
            master = pool.is_master()
        else:
            pool = None
            master = True
        import run_cosmolike_mpp

        params = copy.deepcopy(self.params)
        params['mask_file'] = self.input_path("mask")
        params['source_nz'] = self.input_path("nz_source_text")
        params['lens_nz'] = self.input_path("nz_lens_text")
        params['cov_file'] = self.input_path("cov_txt")
        params['data_file'] = self.input_path("2pt_text")
        if master:
            params['cosmolike_fid_datavector'] = self.output_path('cosmolike_fid_datavector')
        
        run_cosmolike_mpp.main(params, pool=pool)
    
        
