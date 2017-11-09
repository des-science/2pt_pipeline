#############################################################################
# do2ptblinding.py 
#-------------------------------------------------------------------------
# This script will, given an npz file containing sets of cosmological
# parameters, a data file containing unblinded 2pt data (and relevant
# dn/dz info), and cosmosis ini files to be used for generating 2pt functions
# like those of the dataset, pseudo-randomly select a set of cosmological
# paramters, then use cosmosis to generate the relevant 2pt blinding factors,
# apply those to unblinded data, producing a duplicate blinded datafile.
#
# To use on command line call:
#  python blind2pt.py -u <input unblinded data file>
#     -c <npz file containing cosm params>
#     -i <ini file for generating 2pt fns with cosmosis>
#     -s <string seed>
#     -t <outfile tag string>
#
# default arguments:
#  -u : two_pt_cov.fits
#  -c : public_output/blindingcosm_bundle_10.npz
#  -i : gen2pt_forblinding.ini
#  -s : blinded
#  -t : bl
#
# string seed is used to pseudo-randomly select one of the
# parameter sets in the npz file
# 
# For now assumes input data is in format matching two_pt_cov.fits, though
#  we could adapt it to accomodate different file types.
#
# Output file will match input, with a tag appended to the name, and
# an additional header will be added to all 2PT tables
# to indicate that it has been blinded. 
#
#-------------------------------------------------------------------------
# Script maintained by Jessica Muir (jlmuir@umich.edu).
#############################################################################

import numpy as np
import getopt, sys, shutil
from astropy.io import fits
from scipy.interpolate import interp1d
import hashlib
from cosmosis.runtime.config import Inifile
from cosmosis.runtime.pipeline import LikelihoodPipeline

#############################################################################
# Class for bundling cosmology data
#############################################################################

class Cosmology(object):
    """Class containing cosmological params"""
    def __init__(self,omega_m=None,h0=None,omega_b=None,sigma8=None,w0=None,wa=None,bias=None, roundto = 6):
        # Default values chosen to match
        # https://cdcvs.fnal.gov/redmine/projects/des-theory/wiki/Y1_Analysis_Choices
        fid_omega_m = 0.295
        fid_h0 = 0.6881
        fid_omega_b = 0.0468
        fid_sigma8 = 0.8344
        fid_w0 = -1.
        fid_wa = 0.
        fid_bias = 1.65

        if omega_m == None:
            self.omega_m = fid_omega_m
        else:
            self.omega_m = omega_m
        if h0==None:
            self.h0 = fid_h0
        else:
            self.h0= h0
        if omega_b == None:
            self.omega_b = fid_omega_b
        else:
            self.omega_b= omega_b
        if sigma8 == None:
            self.sigma8 = fid_sigma8
        else:
            self.sigma8= sigma8
        if w0 == None:
            self.w0 = fid_w0
        else:
            self.w0= w0
        if wa == None:
            self.wa = fid_wa
        else:
            self.wa= wa
        if bias == None:
            self.bias = fid_bias
        else:
            self.bias = bias

    def pnames(self):
        """
        Return list of strings which this class associates with all the
        parameters it holds. 
        """
        return ['omega_m','omega_b','h0','sigma8','w0','wa','bias']
        
    def get_param(self,pstr):
        if pstr == 'omega_m':
            return self.omega_m
        elif pstr == 'omega_b':
            return self.omega_b
        elif pstr == 'h0':
            return self.h0
        elif pstr == 'sigma8':
            return self.sigma8
        elif pstr == 'w0':
            return self.w0
        elif pstr == 'wa':
            return self.wa
        elif pstr == 'bias':
            return self.bias

    def set_param(self,pstr,val):
        if pstr == 'omega_m':
            self.omega_m = val
        elif pstr == 'omega_b':
            self.omega_b = val
        elif pstr == 'h0':
            self.h0 = val
        elif pstr == 'sigma8':
            self.sigma8 = val
        elif pstr == 'w0':
            self.w0 = val
        elif pstr == 'wa':
            self.wa = val
        elif pstr == 'bias':
            self.bias = val

#############################################################################
# FUNCTIONS FOR READING IN BLINDING FACTORS FROM NPZ FILE
#############################################################################
def read_npzfile(npzfile):
    """ 
    Given the name of a .npz file arrays of cosmological parameters,
    read them in 
    """
    dat = np.load(npzfile)
    outdict={}
    for datatype in dat.files:
        outdict[datatype]=dat[datatype]
    dat.close() #need to close npz files to avoid memory leak
    return outdict

#------------------------------------------------------------------
def get_cosm_forind(bundledict,setind):
    """
    Given dictionary of bundled sets of cosmological parameters and an index 
    for one of those sets, returns a Cosmology object. 
    """
    cosm = Cosmology()
    for p in bundledict.keys():
        cosm.set_param(p,bundledict[p][setind])
    return cosm
#------------------------------------------------------------------
def get_cosm_forseedstr(bundledict,setseed='blinded'):
    """
    Given bundled sets of cosm params in a dictionary and a seed string,
    converts the string to an appropriate index, and then returns a Cosmology
    object corresponding to that index.

    If setseed passed as 'random', chooses an index at random. 
    Otherwise, deterministically translates the string to an index.
    """
    Nsets = bundledict.values()[0].size - 1 #0th set is ref, others are shifted
    if setseed == 'random':
        ind = np.random.randint(Nsets)
    else:
        ind = string_to_ind(setseed, Nsets)

    cosm = get_cosm_forind(bundledict, ind +1)
    return cosm

#------------------------------------------------------------------
def string_to_ind(setseed = 'blinded', Nsets = 10):
    """ 
    Deterministically converts any string to a number in 
    the range [0,Nsets)
    """
    hashstr = hashlib.md5(setseed).hexdigest()
    ind = int(int(hashstr,16)%Nsets)
    return ind

#############################################################################
# FUNCTIONS FOR GENERATING BLINDING FACTORS
#############################################################################
def gen_blindingfactors(refcosm,shiftcosm,inifor2pt='gen2pt_forblinding.ini',nz_file = 'two_pt_cov.fits'):
    """
    Given input Cosmology objects refcosm and shiftcosm, plus cosmosis inifiles,
    runs cosmosis pipeline to generate and save C_ell; inifile values are 
    defaults, cosmology params in refcosm, shiftcosm supercede them. 
    """
    twoptdictlist = []
    for c in [refcosm,shiftcosm]:
        ini=Inifile(inifor2pt)
        pipeline=LikelihoodPipeline(ini)

        # set parameters as desired
        for parameter in pipeline.parameters:
            if parameter == ('cosmological_parameters','omega_m'):
                parameter.start = c.omega_m
            elif parameter == ('cosmological_parameters','h0'):
                parameter.start = c.h0
            elif parameter == ('cosmological_parameters','omega_b'):
                parameter.start = c.omega_b
            elif parameter == ('cosmological_parameters','sigma8_input'):
                parameter.start = c.sigma8
            elif parameter == ('cosmological_parameters','w'):
                parameter.start = c.w0
            elif parameter == ('cosmological_parameters','wa'):
                parameter.start = c.wa
            elif parameter == ('bias','b_g'):
                parameter.start = c.bias
            elif parameter == ('fits_nz','nz_file'):
                parameter = nz_file 

        data = pipeline.run_parameters([])
        twoptdictlist.append(twoptdict_from_datablock(data))
        
    refdict = twoptdictlist[0]
    shiftdict = twoptdictlist[1]
    factordict = get_blindfactors(refdict,shiftdict)
    
    return factordict
#------------------------------------------------------    
def twoptdict_from_datablock(block):
    """
    After running cosmosis to compute 2pt functions, extracts 
    the relevant data from the datablock and returns it in a dictionary.
    """
    nbins = block['galaxy_xi', 'nbin_a']
    for entry in ['galaxy_xi', 'galaxy_shear_xi', 'shear_xi']:
        for bin in ['nbin_a', 'nbin_b']:
            if nbins != block[entry, bin]:
                raise Exception('Inconsistent redshift binning')

    gal_gal_theta     = np.rad2deg(60.0) * block['galaxy_xi', 'theta']
    gal_shear_theta   = np.rad2deg(60.0) * block['galaxy_shear_xi', 'theta']
    shear_shear_theta = np.rad2deg(60.0) * block['shear_xi', 'theta']
    gal_gal_l     =  block['galaxy_cl', 'ell']
    gal_shear_l   =   block['galaxy_shear_cl', 'ell']
    shear_shear_l =  block['shear_cl', 'ell']

    gal_gal_xi      = np.zeros([nbins, nbins, len(gal_gal_theta)])
    gal_shear_xi    = np.zeros([nbins, nbins, len(gal_shear_theta)])
    shear_shear_xip = np.zeros([nbins, nbins, len(shear_shear_theta)])
    shear_shear_xim = np.zeros([nbins, nbins, len(shear_shear_theta)])
    gal_gal_cl      = np.zeros([nbins, nbins, len(gal_gal_l)])
    gal_shear_cl    = np.zeros([nbins, nbins, len(gal_shear_l)])
    shear_shear_cl = np.zeros([nbins, nbins, len(shear_shear_l)])
    for b1 in range(nbins):
        for b2 in range(b1 + 1):
            gal_gal_xi[b1,b2,:]      = block['galaxy_xi', 'bin_{0:}_{1:}'.format(b1 + 1, b2 + 1)]
            gal_gal_xi[b2,b1,:]      = block['galaxy_xi', 'bin_{0:}_{1:}'.format(b1 + 1, b2 + 1)]
            shear_shear_xip[b1,b2,:] = block['shear_xi', 'xiplus_{0:}_{1:}'.format(b1 + 1, b2 + 1)]
            shear_shear_xip[b2,b1,:] = block['shear_xi', 'xiplus_{0:}_{1:}'.format(b1 + 1, b2 + 1)]
            shear_shear_xim[b1,b2,:] = block['shear_xi', 'ximinus_{0:}_{1:}'.format(b1 + 1, b2 + 1)]
            shear_shear_xim[b2,b1,:] = block['shear_xi', 'ximinus_{0:}_{1:}'.format(b1 + 1, b2 + 1)]
            gal_gal_cl[b1,b2,:]      = block['galaxy_cl', 'bin_{0:}_{1:}'.format(b1 + 1, b2 + 1)]
            gal_gal_cl[b2,b1,:]      = block['galaxy_cl', 'bin_{0:}_{1:}'.format(b1 + 1, b2 + 1)]
            shear_shear_cl[b1,b2,:] = block['shear_cl', 'bin_{0:}_{1:}'.format(b1 + 1, b2 + 1)]
            shear_shear_cl[b2,b1,:] = block['shear_cl', 'bin_{0:}_{1:}'.format(b1 + 1, b2 + 1)]
            
        for b2 in range(nbins):
            gal_shear_xi[b1,b2,:]    = block['galaxy_shear_xi', 'bin_{0:}_{1:}'.format(b1 + 1, b2 + 1)]
            gal_shear_cl[b1,b2,:]    = block['galaxy_shear_cl', 'bin_{0:}_{1:}'.format(b1 + 1, b2 + 1)]

    outdict = {'gal_gal_theta':gal_gal_theta, \
               'gal_gal_xi':gal_gal_xi, \
               'gal_shear_theta':gal_shear_theta,\
               'gal_shear_xi':gal_shear_xi, \
               'shear_shear_theta':shear_shear_theta, \
               'shear_shear_xip':shear_shear_xip, \
               'shear_shear_xim':shear_shear_xim, \
               'gal_gal_l':gal_gal_l, \
               'gal_gal_cl':gal_gal_cl, \
               'gal_shear_l':gal_shear_l, \
               'gal_shear_cl':gal_shear_cl, \
               'shear_shear_l':shear_shear_l, \
               'shear_shear_cl':shear_shear_cl }

    return outdict

#------------------------------------------------------
def get_blindfactors(refdict,shiftdict):
    """
    Given two dictionaries containing two point data, 
    takes ratio to get blinding factors. 
    """
    ratiodict = {}
    for key in refdict: #all2ptdat:
        if ('_ell' in key) or ('_theta' in key) or ('_l' in key) :
            #'is x data'
            ratiodict[key] = refdict[key]
        else:
            ratiodict[key] = shiftdict[key]/refdict[key]
    return ratiodict

#############################################################################
# FUNCTIONS FOR BLINDING DATA FILES
#############################################################################
#===============================================================
# Functions specific to fits file format given by Niall in spring 2016
#   Expects shear and number density data with these keys:
#===============================================================
def get_2pttype_for_dictkey(dictkey):
    """
    Convert key in blinding factor dictionary to sets of strings used
    in the fits file to designate which kind of 2pt function is being analyzed.
    """
    if dictkey == 'gal_gal_cl':
        return 'GPF','GPF'
    elif dictkey == 'gal_shear_cl':
        return 'GPF','GEF'
    elif dictkey == 'shear_shear_cl':
        return 'GEF','GEF'
    elif dictkey == 'gal_gal_xi':
        return 'GPR','GPR'
    elif dictkey == 'gal_shear_xi':
        return 'GPR','G+R'
    elif dictkey == 'shear_shear_xip':
        return 'G+R','G+R'
    elif dictkey == 'shear_shear_xim':
        return 'G-R','G-R'

#---------------------------------------     
def get_data_from_dict_for_2pttype(type1,type2,datadict):
    """ 
    Given strings identifying the type of 2pt data in a fits file 
    and a dictionary of 2pt data (i.e. the blinding factors), 
    returns the data from the dictionary matching those types.
    """
    #spectra type codes in fits file, under hdutable.header['quant1'] and quant2
    galaxy_position_fourier = "GPF"
    galaxy_shear_emode_fourier = "GEF"
    galaxy_shear_bmode_fourier = "GBF"
    galaxy_position_real = "GPR"
    galaxy_shear_plus_real = "G+R"
    galaxy_shear_minus_real = "G-R"
    
    if type1==galaxy_position_fourier and type2 == galaxy_position_fourier:
        yfromdict=datadict['gal_gal_cl']
        xfromdict=datadict['gal_gal_l']
    elif (type1==galaxy_shear_emode_fourier and type2 == galaxy_position_fourier) or (type2==galaxy_shear_emode_fourier and type2 == galaxy_position_fourier):
        yfromdict=datadict['gal_shear_cl']
        xfromdict=datadict['gal_shear_l']
    elif (type1==galaxy_shear_emode_fourier and type2 == galaxy_shear_emode_fourier):
        yfromdict=datadict['shear_shear_cl']
        xfromdict=datadict['shear_shear_l']
    elif type1==galaxy_position_real and type2 == galaxy_position_real:
        yfromdict=datadict['gal_gal_xi']
        xfromdict=datadict['gal_gal_theta']
    elif (type1==galaxy_shear_plus_real and type2 == galaxy_position_real) or (type2==galaxy_shear_plus_real and type1 == galaxy_position_real):
        yfromdict=datadict['gal_shear_xi']
        xfromdict=datadict['gal_shear_theta']
    elif (type1==galaxy_shear_plus_real and type2 == galaxy_shear_plus_real):
        yfromdict=datadict['shear_shear_xip']
        xfromdict=datadict['shear_shear_theta']
    elif (type1==galaxy_shear_minus_real and type2 == galaxy_shear_minus_real):
        yfromdict=datadict['shear_shear_xim']
        xfromdict=datadict['shear_shear_theta']
    else:
        print "Spectra type {0:s} - {1:s} not recognized.".format(type1,type2)
    return xfromdict,yfromdict

#---------------------------------------
def sample_dictdat_tomatch_fitsdat(table,dictdata):
    """
    Given table of type fits.hdu.table.BinTableHDU containing 2pt data, 
    and retrieves corresponding data from dictionary (blinding factors), 
    then interpolates dict data to match dimensions of fits file data.
    """
    #props = table.header.keys() #holds things like units, datatype
    #colnames = table.data.names
    #print '-------\n',table.name,'\n-------\n',table.header.keys()
    if not table.header.get('2PTDATA'):
        print "Can't match dict data: this fits table doesn't contain 2pt data. Is named:",table.name
        return
    type1 = table.header['QUANT1']
    type2 = table.header['QUANT2']
    bin1fromfits = table.data['BIN1'] #which bin is quant1 from?
    Nbin1 = np.max(bin1fromfits) #labels start at 1, not 0
    bin2fromfits = table.data['BIN2'] #which bin is quant2 from?
    Nbin2 = np.max(bin2fromfits)
    xfromfits = table.data['ANG']
    xfromdict,yfromdict = get_data_from_dict_for_2pttype(type1,type2,dictdata)
    if (Nbin1 != yfromdict.shape[0]) or (Nbin2 != yfromdict.shape[1]):
        raise ValueError('Data file and blinding factors have different numbers of bins.')

    
    samplednewdat=np.zeros(xfromfits.size)
    for i in xrange(samplednewdat.size):
        bin1ind = bin1fromfits[i]-1
        bin2ind = bin2fromfits[i]-1
        samplednewdat[i]=interp1d(xfromdict,yfromdict[bin1ind,bin2ind,:])(xfromfits[i])
    return samplednewdat

#---------------------------------------
def apply2ptblinding_tofits(factordict, origfitsfile = 'two_pt_cov.fits', outftag = "", outfile = "", justfname = False):
    """ 
    Given the dictionary of one set of blinding factors, 
    the name of a  of a fits file containing unblinded 2pt data, 
    and (optional) desired output file name or tag, 
    multiplies 2pt data in original fits file by blinding factors and 
    saves results (blinded data) into a new fits file. 

    If justfname == True, doesn't do any file manipulation, just returns
    string of output filename. (for testing)
    """
    print 'Applying blinding factor to data in',origfitsfile

    # check whether data is already blinded and whether Nbins match
    for table in fits.open(origfitsfile): #look through tables to find 2ptdata
        if table.header.get('2PTDATA'): 
            if table.header.get('BLINDED'): #check for blinding
                #if entry not there, or storing False -> not already blinded
                raise ValueError('Data is already blinded!')
                return
            # check whether number of bins match
            sample_dictdat_tomatch_fitsdat(table, factordict)
            # ^ will raise ValueError if Nbins is mismatched
            
    # set up output file
    if outfile:
        outfname = outfile
    else:
        if not outftag:
            outftag = 'bl'
        outfname = origfitsfile.replace('.fits','_{0:s}.fits'.format(outftag))

    if not justfname:
        shutil.copyfile(origfitsfile,outfname)

        hdulist = fits.open(outfname,mode='update') #update lets us write over

        # apply blinding factors 
        for table in hdulist: #look all tables
            if table.header.get('2PTDATA'):
                table.data['value'] *= sample_dictdat_tomatch_fitsdat(table, factordict)
                #add new header entry to note that blinding has occured, store hash
                table.header['BLINDED'] = True
        
        hdulist.close() # will save new data to file if 'update' was passed when opened
        print "Stored blinded data in",outfname
    return outfname

#===============================================================
# wrapper
#===============================================================
def do2ptblinding(unblindedfile, cosmfile, inifor2pt, outftag = 'bl', seed='blinded'):
    """
    Given unblinded data file, computes and applies blinding factors. 
    Factors are computed doing [shift cosm 2pt fn]/[ref cosm 2pt fn]
    where cosm parameters are taken from pregenerated
    cosmfile, and the shifted cosmology is selected pseudorandomly 
    using a string seed. 
    """
    # get relevant sets of cosm parameters in form of Cosmology objects
    cosmdict = read_npzfile(cosmfile)
    refcosm = get_cosm_forind(cosmdict,0)
    shiftcosm = get_cosm_forseedstr(cosmdict,seed)

    # run cosmosis to get factors
    factordict = gen_blindingfactors(refcosm,shiftcosm,inifor2pt,unblindedfile)
    #^TODO make sure this looks into proper file for dn/dz, etc
    #  currenlty does not and just looks in the ini file for this

    # apply blinding factors to dat, create output blinded file
    blindedfile = apply2ptblinding_tofits(factordict, origfitsfile = 'public_output/two_pt_cov.fits', outftag = outftag, justfname = False)

    return blindedfile
        
    
##############################################################################
##############################################################################
if __name__=="__main__":
    #default values
    callscript = False # if true calls script specifically as written before, 
                      # otherwise calls according to command line args
                      #  [set to true on command line with --script]
    unblindedfile = 'public_output/two_pt_cov.fits' #contains 2pt data to be blinded
                      # [set on command line with -u --unblindedfile]

    cosmfile = 'public_output/cosm4blinding_Y1-161214.npz'
                      # [via command line: -c <fname> or --cosmfile <fname>]
    inifor2pt = 'gen2pt_forblinding.ini'
                      # [set via command line -i or --inifor2pt]
    seed = 'blinded' # if 'random', selects blinding factor set from factor
                      #  file at random, if string passed, will translate string
                      # deterministically to an index, then use that to select
                      # a  set of blinding factors
                      # [via command line, -s <str> or --seed <str>]
    outftag = 'bl' # string added to blinded file name 
                   # so that output file containing blinded data is
                   #  <input fname>_<outftag>.<input file suffix>.
                   # [via command line, -t <string> or --outftag <str>]

    options,remainder = getopt.getopt(sys.argv[1:],'u:c:i:s:t:',['unblindedfile=','cosmfile=','inifor2pt=','seed=','outftag=','script'])
    #print options
    if ('--script') in options:
        callscript = True
    else:
        for opt, arg in options:
            if opt in ('-u','--unblindedfile'):
                unblindedfile = arg
            elif opt in ('-c','--cosmfile'):
                cosmfile = arg
            elif opt in ('-i','--inifor2pt'):
                inifor2pt= arg
            elif opt in ('-s','--seed'):
                seed = arg
            elif opt in ('-t', '--outftag'):
                outftag = arg

    if callscript:
        print "CALLING SCRIPT"
        # currently is just defaults, can adjust here for testing
        do2ptblinding(unblindedfile = unblindedfile, cosmfile =  cosmfile, seed = seed, inifor2pt = inifor2pt, outftag = outftag)
    else:
        print 'unblindedfile',unblindedfile
        print 'cosmfile',cosmfile
        print 'seed',seed
        print 'inifor2pt',inifor2pt
        print 'outftag',outftag
        do2ptblinding(unblindedfile = unblindedfile, cosmfile =  cosmfile, seed = seed, inifor2pt = inifor2pt, outftag = outftag)
        

