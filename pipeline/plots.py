from __future__ import print_function, division
from .stage import PipelineStage
import matplotlib
matplotlib.use("agg")
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer

latex_names = {
    "cosmological_parameters--omega_b"      : r"\Omega_b",
    "cosmological_parameters--omega_m"      : r"\Omega_m",
    "cosmological_parameters--omega_nu"     : r"Omega_\nu",
    "cosmological_parameters--ommh2"        : r"\Omega_m h^2",
    "cosmological_parameters--omch2"        : r"\Omega_c h^2",
    "cosmological_parameters--ombh2"        : r"\Omega_b h^2",
    "cosmological_parameters--omnuh2"       : r"\Omega_\nu h^2",
    "cosmological_parameters--h0"           : r"h",
    "cosmological_parameters--hubble"       : r"H_0",
    "cosmological_parameters--w"            : r"w",
    "cosmological_parameters--wa"           : r"w_a",
    "cosmological_parameters--omega_k"      : r"\Omega_k",
    "cosmological_parameters--omega_l"      : r"\Omega_\Lambda",
    "cosmological_parameters--tau"          : r"\tau",
    "cosmological_parameters--n_s"          : r"n_s",
    "cosmological_parameters--A_s"          : r"A_s",
    "cosmological_parameters--sigma_8"      : r"\sigma_8",
    "cosmological_parameters--sigma8_input" : r"\sigma_8",
    "cosmological_parameters--r_t"          : r"r_t",
    "cosmological_parameters--yhe"          : r"Y_\mathrm{He}",
    "supernova_params--alpha"               : r"\alpha",
    "supernova_params--beta"                : r"\beta",
    "supernova_params--M0"                  : r"M_0",
    "supernova_params--deltam"              : r"\Delta M",
    "planck--A_ps_100"                      : r"A^100_{\mathrm{PS}}",
    "planck--A_ps_143"                      : r"A^143_{\mathrm{PS}}",
    "planck--A_ps_217"                      : r"A^217_{\mathrm{PS}}",
    "planck--A_cib_143"                     : r"A^143_{\mathrm{CIB}}",
    "planck--A_cib_217"                     : r"A^217_{\mathrm{CIB}}",
    "planck--A_sz"                          : r"A_{\mathrm{SZ}}",
    "planck--r_ps"                          : r"r_{\mathrm{PS}}",
    "planck--r_cib"                         : r"r_{\mathrm{CIB}}",
    "planck--n_Dl_cib"                      : r"r_{\mathrm{PS}}",
    "planck--cal_100"                       : r"{\cal C}_{100}",
    "planck--cal_143"                       : r"{\cal C}_{143}",
    "planck--cal_217"                       : r"{\cal C}_{217}",
    "planck--xi_sz_cib"                     : r"\xi^{\mathrm{SZ}}_{\mathrm{CIB}}",
    "planck--A_ksz"                         : r"A_\mathrm{KSZ}",
    "planck--Bm_1_1"                        : r"B_{11}",
    "planck--A"                             : r"A_{\mathrm{planck}}",
    "bias_parameters--alpha"                : r"\alpha",
    "bias_parameters--b0"                   : r"b_0",
    "bias_parameters--a"                    : r"a",
    "bias_parameters--A"                    : r"A",
    "bias_parameters--c"                    : r"c",
    "post_friedmann_parameters--d_0"        : r"D_0",
    "post_friedmann_parameters--d_inf"      : r"D_\infty",
    "post_friedmann_parameters--q_0"        : r"Q_0",
    "post_friedmann_parameters--q_inf"      : r"Q_\infty",
    "post_friedmann_parameters--s"          : r"s",
    "post_friedmann_parameters--k_c"        : r"k_c",
    "clusters--M_max"                       : r"M_\mathrm{max}",
    "intrinsic_alignment_parameters--A"     : r"A",

}

class Plots(PipelineStage):
    name = "plots"

    inputs = {
        "chain"        : ("cosmology", "chain.txt")          ,
    }

    outputs = {
        "corner" : "corner.png",
    }

    def run(self):
        self.engine=chainconsumer.ChainConsumer()


        self.engine.add_chain(filename, name="1", parameters=names, posterior=post)
        self.engine.configure(sigmas=[0,1,2], kde=False)
        self.engine.plot()

    def read_chain(self):
        filename = self.input_path("chain")
        data = np.loadtxt(filename)
        names = open(filename).readline().strip("#").split()

    def write(self):
        pass
