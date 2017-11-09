from .stage import PipelineStage
import numpy as np
import blind2pt

class Blinding(PipelineStage):
    name = "blind"
    # fixed cosmological parameter sets input.
    # cosmosis parameter file and values file
    # string seed from the parameter
    inputs = {
        "2pt_extended"          : ("2pt", "2pt_extended_data.fits") ,
        "2pt_ng"                : ("2pt", "2pt_NG.fits") ,
        "2pt_g"                 : ("2pt", "2pt_G.fits") ,
    }
    outputs = {
        "2pt_ng"           : "2pt_NG_blinded.fits",
        "2pt_g"            : "2pt_G_blinded.fits",
        "2pt_extended"     : "2pt_extended_data_blinded.fits",
    }

    def __init__(self, param_file):
        super(Blinding,self).__init__(param_file)

    def run(self):
        blinding_string = self.params['blinding_seed']
        cosm_file = self.params['shiftcosm_file']
        inifor2pt = self.params['inifor2pt'] 

        cosmdict = blind2pt.read_npzfile(cosm_file)
        refcosm = blind2pt.get_cosm_forind(cosmdict,0)
        shiftcosm = blind2pt.get_cosm_forseedstr(cosmdict,blinding_string)

        factordict = blind2pt.gen_blindingfactors(refcosm,shiftcosm,inifor2pt,self.input_path('2pt_g'))
        # ^ the only thing this uses from the input file is n(z), so if that's the same for all
        # input files, then you only need to do this once 

        for file_type in ['2pt_g', '2pt_ng', '2pt_extended']:
            input_file = self.input_path(file_type)
            output_file = self.output_path(file_type)
            blind2pt.apply2ptblinding(factordict,input_file, inifor2pt, outfile = output_file)

    def write(self):
        pass
