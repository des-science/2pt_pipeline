import twopoint
import numpy as np
from .stage import PipelineStage
import yaml
import os

class Text2Point(PipelineStage):
    name = "2pt_text"
    inputs = {
        "2pt" : ("2pt_fits", "2pt_NG.fits"),

    }
    outputs = {
        "mask" : "mask.txt",
        "2pt_text"  : "2pt.txt",
        "nz_source_text" : "source_nz.txt",
        "nz_lens_text" : "lens_nz.txt",
    }

    def __init__(self, param_file):
        super(Text2Point,self).__init__(param_file)
        filename = self.input_path('2pt')
        self.data = twopoint.TwoPointFile.from_fits(filename, covmat_name=None)

    def run(self):
        self.nz_to_text()
        self.spectra_to_text()
        self.make_mask_file()

    def write(self):
        pass

    def make_mask_file(self):
        xip = self.data.get_spectrum("xip")
        xim = self.data.get_spectrum("xim")
        ggl = self.data.get_spectrum("gammat")
        wtheta = self.data.get_spectrum("wtheta")

        mask_length = sum(len(s) for s in self.data.spectra)
        masks = []
        for xi in [xip,xim,ggl,wtheta]:
            n1 = xi.bin1.max()
            n2 = xi.bin2.max()
            for i in xrange(n1):
                for j in xrange(n2):
                    theta, data = xi.get_pair(i+1,j+1)
                    if len(theta)==0:
                        continue
                    param = "angle_range_{}_{}_{}".format(xi.name,i+1,j+1)
                    theta_min, theta_max = self.params[param]
                    mask = (theta>theta_min) & (theta<theta_max)
                    masks.append(mask.astype(int))
        mask = np.concatenate(masks)
        assert mask_length == len(mask), "Total data vector length = {}. Constructed mask length = {}".format(mask_length, len(mask))

        filename = self.output_path("mask")
        f = open(filename, 'w')
        for i,m in enumerate(mask):
            f.write("{} {}\n".format(i,m))
        f.close()

    def nz_to_text(self):
        # This bit actually gets done twice, once in the nofz
        # stage so that the compute_covariance has what it needs,
        # and once here, so that if we want to use this code more
        # generally then we still save nz.  It can't hurt and 
        # the files should be the same.
        nz_pos = self.data.get_kernel("nz_lens")
        nz_shape = self.data.get_kernel("nz_source")

        lens_cols = [nz_pos.zlow] + nz_pos.nzs
        source_cols = [nz_shape.zlow] + nz_shape.nzs

        np.savetxt(self.output_path("nz_lens_text"), np.transpose(lens_cols))
        np.savetxt(self.output_path("nz_source_text"), np.transpose(source_cols))

    def spectra_to_text(self):
        output = open(self.output_path("2pt_text"), "w")

        #TODO: Handle missing spectra.
        xip = self.data.get_spectrum("xip")
        xim = self.data.get_spectrum("xim")
        ggl = self.data.get_spectrum("gammat")
        wtheta = self.data.get_spectrum("wtheta")

        row = 0
        for xi in [xip,xim,ggl,wtheta]:
            n1 = np.unique(xi.bin1)
            n2 = np.unique(xi.bin2)
            for i in n1:
                for j in n2:
                    theta, data = xi.get_pair(i,j)
                    for d in data:
                        output.write("{} {}\n".format(row, d))
                        row += 1
                        
