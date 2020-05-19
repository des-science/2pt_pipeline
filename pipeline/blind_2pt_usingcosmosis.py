""""
blind_2pt_usingcosmosis.py
--------------------------------------------------------------------------------
This script will apply a blinding factor to 2pt functios stored in a fits
file, using cosmosis to compute the blinding factors.

The workhorse function that puts everything together is do2ptblinding().
This script can be called with command line arguments, or by calling that
function in another script.

The script uses cosmosis, so you'll need to  source its setup file before
running this.
............................................................
Arguments:
  -u, --origfits : name of unblinded fits file
  -i, --ini : ini file containing template for generating 2pt functions. Should
                -> use a binning that matches that of the origfits file
                -> Reference a values file centered at desired reference
                   cosmology
  -s, --seed: string used to seed parameter shift selection
  -t, --bftype: blinding factor type. Can be 'add', 'mult', or
                     'multNOCS' (mult with no cov scaling) [MULT OPTION IS DISABLED]
  -o, --outfname : output filename; only set for testing,  default behavior is
               outfname = [origfits]_[seed].fits
  -t, --outftag : string to label output filename, for testing purposes. If set
                  when outfname is None (default), then
                  outfname = [origfits]_[outftag]_[seed].fits
  -m, --paramshiftmodule : string string of a python filename (minus the .py)
                  which can be used to import a function to draw parameter
                   shifts (see draw_paramshift for more info)
  --seedinfname: if True, appends seed to blinded data filename. Default is False. 
  --seedinfits: if True, stores seed in KEYWORD entry in blinded fits file. Default True
............................................................

What it does:
1. Using a string seed, pseudo-randomly draw a shift in parameters. This
   will eitherbe drawn from a flat distribution in some predefined parameter
   ranges, or from  a distribution defined in paramshiftmodule. Return
   dictionary of shifts where keys are parameter names matching those expected
   by run_cosmosis_togen_2ptdict.
   -> See draw_paramshift

2. Using a cosmosis parameter ini file template, compute 2pt functions at
   reference and shifted cosmologies by running the cosmosis  twice. This
   should work no matter what sampler shows up in the template ini file.
   (If it uses the test sampler,  make sure the output is nothing, so that
   it doesn't save the cosmology and 2pt info.) Gets the 2pt functions into
   dictionaries of a specific format.
   -> See run_cosmosis_togen_2ptdict

4. Take ratio or difference of 2pt datavectors to get blinding factor, in same
   dictionary format. (NOTE, MULTIPLICATIVE BLINDING IS CURRENTLY DISABLED)
   -> See get_factordict

5. Apply blinding factor to desired datafile, saving a new fits file with
   string key in filename.
   -> See apply2ptblinding_tofits
--------------------------------------------------------------------------------
Script maintained by Jessica Muir (jlmuir@stanford.edu).
"""
from __future__ import print_function, division
import numpy as np
import getopt, sys, shutil, os
from astropy.io import fits
import hashlib
import importlib
import time
from scipy.interpolate import interp1d

# pull type table and some other from the cosmosis 2pt likelihood dir
# This will only work if cosmosis environment has been set up
csd = os.environ['COSMOSIS_SRC_DIR'] + '/'
typefilepath = csd + 'cosmosis-standard-library/likelihood/2pt/twopoint_cosmosis.py'
# get dir and file name
dirname, filename = os.path.split(typefilepath)
# split off .py
impname, ext = os.path.splitext(filename)
# add directory to path
sys.path.insert(0, dirname)
# import the library
twopoint_cosmosis = __import__(impname)
from twopoint_cosmosis import type_table


#############################################################################
# FUNCTIONS FOR GETTING SHIFT IN PARAMETER SPACE
#############################################################################

DEFAULT_PARAM_RANGE = {'cosmological_parameters--sigma8_input':(0.834-3*.04,0.834+3*0.04),\
                       'cosmological_parameters--w':(-1.5,-.5)}
HARD_CODED_BLINDING = 'I_desperately_need_coffee' #'testblinding'  #keyword is the one truth that unites us all


def draw_paramshift(seedstring='blinded', ranges = DEFAULT_PARAM_RANGE,\
                    importfrom = None):
    """
    Given a string seed, pseudo-randomly draws shift in parameter space.

    By default, draws random values for all the parameters with ranges defined
    in the dictionary 'ranges'. Note:
      - parameter names should match those used by cosmosis (see
        DEFAULT_PARAM_RANGE dict for an example)
      - ranges is a dictionary with parameter names as keys, the values are
        tuples set up as (min,max). The parameter will be drawn from a flat
        distribution between min and max.

    importfrom:
    If you'd like to draw from a different distribution, create another .py
    file with an function in it called draw_paramshift, making sure it return
    a dictuionary with the same format as below:
      - If you pass a string with that file's name as an argument importfrom,
        that distribution will be used instead of this one.
      - The returned dictionary will be expected to have key 'SHIFTS' associated
        with a bool to determine how the values in that dictionary are used.
           pdict['SHIFTS'] = False --> p_shift = pdict['paramname']
           pdict['SHIFTS'] = True  --> p_shift = p_ref + pdict['paramname']

    Make sure any parameters that are shifted here have names matching
    how cosmosis uses them, so that the code in the for loop starting with
    'for parameter in pipeline.parameters' in run_cosmosis_togen_2ptdict
    will work.
    """
    #do something convoluted to turn the string into a number that can be
    # used to seed numpy.random
    seedind = int(int(hashlib.md5(seedstring.encode('utf-8')).hexdigest(),16)%1.e8)
    np.random.seed(seedind)
    #do the paramter shifts
    if importfrom is None:
        params2shift = sorted(ranges.keys()) #sort to make sure it's always the same order
        Nparam = len(params2shift)
        shiftfrac = np.random.rand(Nparam) #array of values between 0 and 1
        mins = np.array([ranges[k][0] for k in params2shift])
        maxs = np.array([ranges[k][1] for k in params2shift])
        dparams = mins + (maxs - mins)*shiftfrac
        pdict = {params2shift[i]:dparams[i] for i in range(Nparam)}
        pdict['SHIFTS'] = False #set to false since it's parameter values
                                # stored in the dictionary, not changes
    else:
        paramdraw = importlib.import_module(importfrom)
        pdict = paramdraw.draw_paramshift()
        if 'SHIFTS' not in list(pdict.keys()):
            raise ValueError("Can't find key 'SHIFTS' in pdict. Set to True if the dict contains Delta params, False if it contains param values.")

    return pdict

#############################################################################
# FUNCTIONS FOR GETTING BLINDING FACTORS
#############################################################################


# Generate 2pt funcs for set of params
def run_cosmosis_togen_2ptdict(pdict={},inifile='pipeline/blinding_params_template.ini',nz_file = None, angles_file = None):
    """
    Runs cosmosis pipeline to generate 2pt functions.

    If a parameter value is passed through the dictionary pdict...
        ...and SHIFTS = False, the param it will be set to that value.
        ...and SHIFTS = True, the param will be shifted by that amount
           from the number in the values file
    Otherwise, it will be set to whatever value is in values file.

    Note that you'll need to source the cosmosis setup file before running this.

    The nz_file and angles_file variables: WARNING: USE WITH CAUTION.
    These are optional strings pointing to fits files where 
    to get n(z) distributions (nz_file) or theta values (angles_file). 
    >>  will only work with set-up matching module labels and set-up 
    >>  These entries are here to accommodate some quirks in how  DES Y3 2pt 
    measurement pipeline works. The default behavior  is that the theory
    calculations will be done following whatever fits files are listed in 
    the template cosmosis in file. This ini file should have the same settings
    you would be using to parameter estimation. These string variables 
    can be sued if you want to pull n(z)'s or angle values from other files. 
    Use with caution! The names this feature uses to change filename settings
    in the cosmosis datablock may be  specific to the DES Y3
    2pt pipeline, and using this feature will make it harder to check
    that you're blinding using the same modeling pipeline as for parameter estimation.
    """
    from cosmosis.runtime.config import Inifile
    from cosmosis.runtime.pipeline import LikelihoodPipeline
    from cosmosis.datablock.cosmosis_py import block
    ini=Inifile(inifile)
    print('     cosmosis ini object initialized from',inifile)
    #ini.set('fits_nz','nz_file',unblfile)

    # change some settings to make sure we don't accidently ouptut info
    if 'test' in ini.__dict__['_sections'].keys():
        ini.__dict__['_sections']['test']['save_dir']=''
    if 'output' in ini.__dict__['_sections'].keys():
        ini.__dict__['_sections']['output']['filename']=''
    ini.__dict__['_sections']['pipeline']['debug']='F'
    ini.__dict__['_sections']['pipeline']['quiet']='T'

    # a structure like what follows could be used to make sure
    # angles and n(z) are being used consistently with file being blinded
    # as in, make sure that they are referencing it for angles and n(z).
    # However, this isn't really ideal, as how it is set up will be specific
    #   to the 3x2pt pipeline and may cause issues when/if the same script
    #   is bein used for 5x2pt or other observables. Really the template ini
    #   file should handle these choices, and hsould be set up in the same way
    #   that you would set up an ini file to do parameter estimation on your
    #   data file. 
    if angles_file is not None:  #fits file to get theta ranges from
        hadsections=[]
        for section in ['shear_2pt_eplusb','shear_2pt_eminusb','2pt_gal','2pt_gal_shear']:
            if section in ini.__dict__['_sections'].keys():
                ini.__dict__['_sections'][section]['theta_file']=angles_file
                hadsections.append(section)
        print("CHANGED theta_file TO ",angles_file," FOR:",hadsections," \n IF YOU MEANT TO CHANGE IT FOR OTHER 2PT FUNCTIONS, SOMETHING WENT WRONG.")
                
    if nz_file is not None:   # fits file to get n(z) from
        section = 'fits_nz'
        if section in ini.__dict__['_sections'].keys():
            ini.__dict__['_sections'][section]['nz_file']=nz_file
        else:
            raise ValueError("You specified nz_file as "+nz_file+", but I can't find the fits_nz module settings in  your ini file.")


    pipeline = LikelihoodPipeline(ini)
    print('     cosmosis pipeline object initialized')


    doshifts = len(list(pdict.keys()))
    if doshifts:
        SHIFTS = pdict['SHIFTS']
        haveshifted = []
        # set parameters as desired
    for parameter in pipeline.parameters:
        key = str(parameter)

        #set the values
        if doshifts:
            try:
                if SHIFTS:
                    parameter.start = parameter.start + pdict[key]
                else:
                    parameter.start = pdict[key]
                haveshifted.append(key)
            except:
                #print '  no entry in pdict'
                pass

        # need to set all of the parameters to be fixed for run_parameters([])
        #  to work. Doing this will effectively run things like the test sampler
        #  no matter what sampler is listed in the ini file
        pipeline.set_fixed(parameter.section, parameter.name, parameter.start)

    if doshifts:
        #print('haveshifted',haveshifted)
        #print('pdictkeys',pdict.keys())
        pass

    if doshifts and (len(haveshifted)!= len(list(pdict.keys())) -1):
        raise ValueError("WARNING: YOU ASKED FOR SHIFTS IN PARAMTERS NOT IN THE COSMOSIS PIPELINE.")
        #print("  asked for shifts in:",list(pdict.keys()))
        #print("  did shifts in:",haveshifted)

    # run pipeline to generate 2pt functions for that cosmology
    print("RUNNING PIPELINE=============== [{0:s}]".format(time.strftime("%Y-%m-%d %H:%M")))
    data = pipeline.run_parameters([])
    print(' at end of run_cosmosis_togen_datavec [{0:s}]'.format(time.strftime("%Y-%m-%d %H:%M")))
    twoptdict =  get_twoptdict_from_pipelinedata(data)

    return twoptdict

#============================================================
def get_bin_pairs(bin1,bin2):
    """
    This is pulled from cosmosis-standard-library/likelihood/2pt/twopoint.py
    """
    unique_pairs = []
    for p in zip(bin1,bin2):
        if p not in unique_pairs:
            unique_pairs.append(p)
    return unique_pairs

def get_twoptdict_from_pipelinedata(data):
    """
    Given datablock object returned by cosmosis pipeline.run_parameters([]),
    get twopt data in a dictionary in the format expected by other functions
    in this script.

    For auto spectra where z bin label order doesn't matter, both orders
    will be stored. (e.g. the same data will be there for bins 1-2 and 2-1).

    """
    #type_table #keys are (type1,type2), values are (section, x, y)
    # get list of possible sections
    outdict={}
    for k in type_table.keys():
        section,  xlabel, binformat= type_table[k]
        # xlabel is either ell or theta
        # binformat will be bin_{0}_{1} or similar
        if data.has_section(section):
            types = k
            xkey,ykey = get_dictkey_for_2pttype(types[0],types[1])
            x,y,bins, is_binavg, x_mins, x_maxs = spectrum_array_from_block(data,section,types, xlabel, binformat)
            outdict[xkey]=x
            outdict[ykey]=y
            outdict[ykey+'_bins'] = bins
            outdict[ykey+'_binavg']=is_binavg
            if is_binavg:
                outdict[xkey+'_mins'] = x_mins
                outdict[xkey+'_maxs'] = x_maxs

    return outdict


#-----------------------------------------------
#def spectrum_array_from_block(block, section_name, types, real_space, bin_format = 'bin_{0}_{1}'):
def spectrum_array_from_block(block, section_name, types, xlabel='theta', bin_format = 'bin_{0}_{1}'):
    """
    >> Updated 4/21/20 to work with bin averaging implemented for DES Y3
    
    Initially adapting this from save_2pt.py and 2pt_like code.

    No scale cutting implemented or angular sampling;
    that will be handled later. Since we
    track bin numbers and angle values, which will be checked against
    the unblinded fits file in get_dictdat_tomatch_fitsdat.
    """
    if xlabel=='theta':
        is_binavg = block[section_name,"bin_avg"]
    else:
        is_binavg = False #no bin averaging in fourier space

    # for cross correlations we must save bin_ji as well as bin_ij.
    # but not for auto-correlations.
    is_auto = (types[0] == types[1])
    if block.has_value(section_name, "nbin"):
        nbin_a = block[section_name, "nbin"]
    else:
        nbin_a = block[section_name, "nbin_a"]
        nbin_b = block[section_name, "nbin_b"]


    #This is the ell/theta values that have been calculated by cosmosis,
    # if bin averaging, these should match what's in the fits files (up to rounding)
    # if interpolating, angles will be more densely sampled than values in fits files.
    theory_angle = block[section_name,xlabel]
    n_angle = len(theory_angle) #whatevers in the block, length of array
    if is_binavg:
        theory_angle_edges = block[section_name,xlabel+'_edges'] #angle bin edges

    #The fits format stores all the measurements
    #as one long vector.  So we build that up here from the various
    #bins that we will load in.  These are the different columns
    value = []
    bin1 = []
    bin2 = []
    angles = []
    # these will only be used if averaging over angular bins
    angle_mins = []
    angle_maxs = []

    # n.b. don't have suffix option implemented here

    #Bin pairs. Varies depending on auto-correlation
    for i in range(nbin_a):
        if is_auto:
            jmax = i+1
        else:
            jmax = nbin_b
        for j in range(jmax):
            #Load and interpolate from the block
            binlabel = bin_format.format(i+1,j+1)
            if block.has_value(section_name, binlabel):
                cl = block[section_name, binlabel]
                bin1.append(np.repeat(i + 1, n_angle))
                bin2.append(np.repeat(j + 1, n_angle))
                angles.append(theory_angle)
                if is_binavg:
                    angle_mins.append(theory_angle_edges[:-1])
                    angle_maxs.append(theory_angle_edges[1:])
                value.append(cl)

                if is_auto and i!=j: #also store under flipped z bin labels
                    # this allows the script to work w fits files uing either convention
                    bin1.append(np.repeat(j + 1, n_angle))
                    bin2.append(np.repeat(i + 1, n_angle))
                    value.append(cl)
                    angles.append(theory_angle)
                    if is_binavg:
                        angle_mins.append(theory_angle_edges[:-1])
                        angle_maxs.append(theory_angle_edges[1:])
                    

    #Convert all the lists of vectors into long single vectors
    value = np.concatenate(value)
    bin1 = np.concatenate(bin1)
    bin2 = np.concatenate(bin2)
    bins = (bin1,bin2)
    angles = np.concatenate(angles)
    if is_binavg:
        angle_mins = np.concatenate(angle_mins)
        angle_maxs = np.concatenate(angle_maxs)
    else:
        angle_mins = None
        angle_maxs = None

    return angles, value, bins, is_binavg, angle_mins, angle_maxs

#-----------------------------------------------
class SpectrumInterp(object):
    """
    This is copied from 2pt_like, for get_data_from_dict_for_2pttype

    Should not get used if bin averaging is being used; is in place
    for if theory vector in datablock is very densely sampled, this 
    gets used to pick out values corresponding to desired angle positions
    """
    def __init__(self,angle,spec,bounds_error=True):
        if np.all(spec>0):
            self.interp_func=interp1d(np.log(angle),np.log(spec),bounds_error=bounds_error,fill_value=-np.inf)
            self.interp_type='loglog'
        elif np.all(spec<0):
            self.interp_func=interp1d(np.log(angle),np.log(-spec),bounds_error=bounds_error,fill_value=-np.inf)
            self.interp_type='minus_loglog'
        else:
            self.interp_func=interp1d(np.log(angle),spec,bounds_error=bounds_error,fill_value=0.)
            self.interp_type="log_ang"

    def __call__(self,angle):
        if self.interp_type=='loglog':
            spec=np.exp(self.interp_func(np.log(angle)))
        elif self.interp_type=='minus_loglog':
            spec=-np.exp(self.interp_func(np.log(angle)))
        else:
            assert self.interp_type=="log_ang"
            spec=self.interp_func(np.log(angle))
        return spec


#============================================================
# ONLY USED FOR MULT BLINDING, SO COMMENTING OUT
# def get_twoptdict_fromfits_1darrays(fitsfile):
#     """
#     Given  fits files containing two point functions, read in data.

#     Returns a dictionary with keys specified in get_2pttype_fordictkey
#     with angle and 2pt data in entries specified by xkey and ykey.
#     """
#     outdict = {}
#     # for indexing: [which 2pt func][which input file]
#     # for each file there will be a dictionary where tuples (bin1,bin2) are keys
#     h = fits.open(fitsfile)
#     for table in h:
#         if table.header.get('2PTDATA'):
#             t = table
#             type1 = t.header['QUANT1']
#             type2 = t.header['QUANT2']

#             galaxy_position_fourier = "GPF"
#             galaxy_shear_emode_fourier = "GEF"
#             galaxy_shear_bmode_fourier = "GBF"
#             galaxy_position_real = "GPR"
#             galaxy_shear_plus_real = "G+R"
#             galaxy_shear_minus_real = "G-R"
#             cmb_kappa_real = "CKR"
            
#             xkey,ykey = get_dictkey_for_2pttype(type1,type2)
            
#             xgrid = t.data['ANG']
#             ygrid = t.data['VALUE']
#             bin1 = t.data['BIN1']
#             print('\n\n\n JUST GOT THE VALUE \n\n\n')
#             bin2 = t.data['BIN2']
#             angbins = t.data['ANGBIN']
#             outdict[xkey] = xgrid
#             outdict[ykey] = ygrid
#             outdict[ykey+'_bins'] = (bin1,bin2)
#             #outdict[ykey+'_angbins'] = angbins

#             #>>>NEW
#             #for bin average stuff
#             if "ANGLEMIN" in t.data.names:
#                 xkeymin = xkey+'_min'
#                 outdict[xkeymin]= t.data['ANGLEMIN']
#             if "ANGLEMAX" in t.data.names:
#                 xkeymax = xkey+'_max'
#                 outdict[xkeymax]= t.data['ANGLEMAX']
#     h.close()
#     return outdict
#============================================================
def get_factordict(refdict,shiftdict,bftype='add'):
    """
    Given two point dictionaries for reference and shifted cosmology,
    returns dictionary of blinding factors.

    bftype = what kind of blinding factor is it?
             'add' : bf = - ref + shift
             'mult': bf = shift/ref
    """
    factordict = {}
    for key in refdict:
        end = key[key.rfind('_')+1:]
        #print(key,end)
        # don't take ratios or differences of angle/multipole info
        if end in ['ell','l','theta','bins','angbins','binavg','mins','maxs']:
            #print '    no change'
            factordict[key] = refdict[key]
        else:
            if bftype=='mult' or bftype=='multNOCS':
                #print '    dividing!'
                factordict[key] = shiftdict[key]/refdict[key]
            elif bftype=='add':
                #print '    adding'
                factordict[key] = shiftdict[key] - refdict[key]
            else:
                raise ValueError('In get_factordict: blinding factor type not recognized')
    return factordict

#===============================================================
#   HELPER FUNCTIONS FOR WORKING WITH DICTS AND FITS FILES
#  Expects shear and number density data with specific keys. If
#  we want to extend this script to handle other summary statistics
#  we'd need to add appropriate keys to these next few functions.
#===============================================================
def get_2pttype_for_dictkey(dictkey):
    """
    Convert key in blinding factor dictionary to sets of strings used
    in the fits file to designate which kind of 2pt function is being analyzed.
    (this script's naming -> fits file 2pt table naming)
    """
    for k in type_table.keys():
        if type_table[k][0]==dictkey:
            return k

def get_dictkey_for_2pttype(type1,type2):
    """
     Convert strings used in fits file to label spectra type in fits file to
     dictionary keys expected by this script's functions.
     (fits file 2pt table naming -> this script's naming)
    """
    firstdict = {'G':'galaxy', 'C':'cmb' }
    seconddict = {'P':'position','E':'shear_emode','B':'shear_bmode','+':'shear_plus','-':'shear_minus','K':'kappa'}
    thirddict = {'R':'real', 'F':'fourier' }
    
    if type1 in ("GPF","GEF","GBF","GPR","G+R","G-R","CKR","GPF","GEF","GBF"):
        #translate short codes into longer strings
        newtypes = ['_'.join([firstdict[t[0]],seconddict[t[1]],thirddict[t[2]]]) for t in [type1,type2]]
        type1 = newtypes[0]
        type2 = newtypes[1]           

    try:
        section, xlabel, ylabel = type_table[(type1,type2)]

        ykey = section
        xkey = section+'_'+xlabel
    except:
        raise ValueError("Spectra type not recognized: {0:s}, {1:s} ".format(type1,type2))

    return xkey,ykey
#---------------------------------------
def get_data_from_dict_for_2pttype(type1,type2,bin1fits,bin2fits,xfits,datadict,fits_is_binavg=True, xfits_mins = None, xfits_maxs = None):
    """
    Given info about 2pt data in a fits file (spectra type, z bin numbers,
    and angular bin numbers, extracts 2pt data from a dictionary of
    e.g. blinding factors to match, with same array format and bin ordering
    as the fits file data.
    """
    xkey,ykey = get_dictkey_for_2pttype(type1,type2)

    is_binavg = datadict[ykey+'_binavg']
    # this check is probably unnecessary, since the data is always bin averaged
    # if dict_is_binavg != fits_is_binavg:
    #     raise ValueError("Theory calc and fits file aren't consistent in whether they do bin averaging vs interpolation for 2pt calculations.")
    if is_binavg and ((xfits_mins is None) or (xfits_maxs is None)):
        raise ValueError("Fits file is bin-averaged but I couldn't find theta values for bin edges.")

    if 'theta' in xkey: #if realspace, put angle data into arcmin
        xmult = 60.*180./np.pi # change to arcmin
    else:
        xmult = 1. #fourier space
        
    xfromdict = datadict[xkey]*xmult #in arcmin (is in radians in datablock)
    xfromdict = xfromdict*xmult
    if is_binavg:
        xfromdict_mins = datadict[xkey+'_mins']*xmult
        xfromdict_maxs = datadict[xkey+'_maxs']*xmult
    yfromdict = datadict[ykey]
    binsdict = datadict[ykey+'_bins']
    b1dict = binsdict[0]
    b2dict = binsdict[1]

    #if get theory calcs in same format as fits ones
    Nentries = bin1fits.size
    yout = np.zeros(Nentries)
    #this structure will store interp functions for b1-b2 combos
    # so we don't have to keep recreating them
    if not is_binavg:
        Nb1fits = max(bin1fits)
        Nb2fits = max(bin2fits)
        interpfuncs = [[None for b2f in range(Nb2fits)]\
                       for b1f in range(Nb1fits)]

    for i in range(Nentries):
        b1 = bin1fits[i]
        b2 = bin2fits[i]
        if is_binavg:
            wherebinsmatch = (b1==b1dict)*(b2==b2dict)

            roundto=4 #round for matching, since there are more decimals in fits than datablock
            wherexmatch_mins = np.around(xfits_mins[i],roundto)==np.around(xfromdict_mins,roundto)
            wherexmatch_maxs = np.around(xfits_maxs[i],roundto)==np.around(xfromdict_maxs,roundto)
            wherexmatch = wherexmatch_mins*wherexmatch_maxs
            #wherexmatch = np.around(xfits[i],roundto)==np.around(xfromdict,roundto)                     
            whichinds = wherebinsmatch*wherexmatch
            howmany = np.sum(whichinds)
            if howmany==0: #no matches
                raise ValueError("No theory calc match for data point: {0:s}, z bins = ({1:d},{2:d}), theta between [{3:0.2f},{4:0.2f}]".format(ykey,b1,b2,xfits_mins[i],xfits_maxs[i]))
            elif howmany>1: #duplicate matches
                raise ValueError("More than one theory calc match for data point: {0:s}, z bins = ({1:d},{2:d}), theta between [{3:0.2f},{4:0.2f}]".format(ykey,b1,b2,xfits_mins[i],xfits_maxs[i]))
                
            yout[i] = yfromdict[whichinds]
        else:
            whichinds = (b1==b1dict)*(b2==b2dict)#*(ab==angbdict)
            # get x and y info for that bin combo
            tempx = xfromdict[whichinds]
            tempy = yfromdict[whichinds]
            if interpfuncs[b1-1][b2-1] is None: #no interpfunc yet, set it up
                yinterp = SpectrumInterp(tempx,tempy)
                interpfuncs[b1-1][b2-1] = yinterp
            else:
                yinterp = interpfuncs[b1-1][b2-1]
            yout[i] =  yinterp(xfits[i])
            # We're returning the y data from the dictionary's array
            # interpolated to match the theta values in the fits file.
    
    return yout

#---------------------------------------
def get_dictdat_tomatch_fitsdat(table,dictdata):
    """
    Given table of type fits.hdu.table.BinTableHDU containing 2pt data,
    retrieves corresponding data from dictionary (blinding factors).

    Expects that same z and theta bin numbers correspond to the same
    z and theta values (i.e. matches up bin numbers but doesn't do
    any interpolation). Theta values will be checked, but z values won't.
    """
    if not table.header.get('2PTDATA'):
        print("Can't match dict data: this fits table doesn't contain 2pt data. Is named:",table.name)
        return
    type1 = table.header['QUANT1']
    type2 = table.header['QUANT2']

    bin1 = table.data['BIN1'] #which bin is quant1 from?
    bin2 = table.data['BIN2'] #which bin is quant2 from?

    # check for bin averaging
    if "ANGLEMIN" in table.data.names:
        fits_is_binavg = True
        xfromfits_mins =  table.data['ANGLEMIN']
        xfromfits_maxs =  table.data['ANGLEMAX']
    else:
        xfromfits_mins = None
        xfromfits_maxs = None
    xfromfits = table.data['ANG']

    yfromdict = get_data_from_dict_for_2pttype(type1,type2,bin1,bin2,xfromfits,dictdata,fits_is_binavg=fits_is_binavg,xfits_mins = xfromfits_mins, xfits_maxs = xfromfits_maxs)
    
    return yfromdict

#############################################################################
# APPLY BLINDING FACTORS TO FITS FILE
#############################################################################

def apply2ptblinding_tofits(factordict, origfitsfile = 'two_pt_cov.fits', outfname = None, outftag = "_BLINDED", justfname = False,bftype='add',storeseed='notsaved'):
    """
    Given the dictionary of one set of blinding factors,
    the name of a  of a fits file containing unblinded 2pt data,
    and (optional) desired output file name or tag,
    multiplies 2pt data in original fits file by blinding factors and
    saves results (blinded data) into a new fits file.

    If argument is passed for outfname, that will be the name of output file,
    if not, will be <input filename>_<outftag>.fits. The script will check
    that the outfname is different than the original, unblinded file. If they
    match, it will revert to default behavior for naming the blinded file
    so that the unblinded file isn't overwritten.
    > if storeseed=True, will save whatever string is there to entry in fits file

    If justfname == True, doesn't do any file manipulation, just returns
    string of output filename. (for testing)

    bftype can be 'add','mult' or 'mult-nocs'. For data vec d, blinding factor f
        'add' - do additive blinding:
                    d_blind = d_input + f, f = d_shift - d_ref, cov_bl = cov
        'mult' - do multiplicative blinding, scale covmat in blinded file:
                    d_blind = d_input*f, f = d_shift/d_ref, cov_bl = f^T*cov*f
        'multNOCS' - do multiplicative blinding, but without covariance scaling
                    d_blind = d_input*f, f = d_shift/d_ref, cov_bl = cov
    """
    print('apply2ptblinding for',origfitsfile)
    print('bftype',bftype)
    # check whether data is already blinded and whether Nbins match
    for table in fits.open(origfitsfile): #look through tables to find 2ptdata
        if table.header.get('2PTDATA'):
            if table.header.get('BLINDED'): #check for blinding
                #if entry not there, or storing False -> not already blinded
                raise ValueError('Data is already blinded!')
                return

    # set up output file
    if outfname == None or outfname==origfitsfile:
        #make sure you can't accidentaly overwrite the original
        #if output filename isn't given, add outftag onto the name of the
        #  unblinded file
        if outftag == None or outftag =='':
            outftag = 'BLINDED-{0:s}-defaulttag'.format(bftype)
        outfname = origfitsfile.replace('.fits','{0}.fits'.format(outftag))
        #outfname = origfitsfile.replace('.fits','_{0:s}.fits'.format(outftag))

    if not justfname:
        shutil.copyfile(origfitsfile,outfname)

        hdulist = fits.open(outfname,mode='update') #update lets us write over
        covscaledict = {}
        #apply blinding factors
        for table in hdulist: #look all tables
            if table.header.get('2PTDATA'):
                factor = get_dictdat_tomatch_fitsdat(table, factordict)

                if bftype=='mult' or bftype=='multNOCS':
                    #print 'multiplying!'
                    table.data['value'] *= factor
                    if bftype=='mult': #store info about how to scale covmat
                        type1 = table.header['QUANT1']
                        type2 = table.header['QUANT2']
                        raise ValueError('Not currently set up to do covariance scaling needed for multiplicative blinding.')

                        # ONLY USED FOR MULT BLINDING, SO COMMENTING OUT
                        # cname = get_covname_for2pttype(type1,type2)
                        # covscaledict[cname] = factor
                        # print('added entry to covscaledict:',cname)
                elif bftype=='add':
                    #print 'adding!'
                    table.data['value'] += factor
                else:
                    raise ValueError('bftype {0:s} not recognized'.format(bftype))

                #add new header entry to note that blinding has occurred, and what type
                table.header['BLINDED'] = bftype
                table.header['KEYWORD'] = storeseed

        # ONLY USED FOR MULT BLINDING, SO COMMENTING OUT
        #print 'covscaledict is',covscaledict
        # if bftype=='mult': #doc cov scaling
        #     covmatname = "COVMAT"
        #     table = hdulist[covmatname]
        #     Ndat = table.data.shape[0] #length of datavector

        #     #get covmat info:
        #     covmat = table.data
        #     headerkeys = list(table.header.keys())
        #     names = []
        #     startinds =[]
        #     for i in range(len(headerkeys)):
        #         key = headerkeys[i]
        #         if key[:5] == 'STRT_':
        #             startinds.append(table.header[key])
        #         if key[:5] == 'NAME_':
        #             name = table.header[key]
        #             names.append(name)
        #     startinds.append(Ndat)

        #     bf = np.zeros(Ndat)
        #     #print names,startinds
        #     for i,covname in enumerate(names):
        #         starti = startinds[i]
        #         endi = startinds[i+1]
        #         bfi = covscaledict[covname]
        #         #print i,covname, endi-starti,bfi.shape
        #         bf[starti:endi] = bfi
        #         # Do cov_new[i,j] = cov[i,j]*bf[i]*bf[j]
        #     table.data = bf.reshape((bf.size,1))*table.data*bf

        hdulist.close() # will save new data to file if 'update' was passed when opened
        print(">>>>Stored blinded data in",outfname)
    return outfname

#############################################################################
# WRAPPER: Put everything together
#############################################################################
def do2ptblinding(seedstring,initemplate,unblindedfits, outfname = None, outftag = "_BLINDED",bftype='add',paramshift_module = None, seedinfname=False, seedinfits = True,nz_file = None, angles_file = None):
    """
    This is the function that gets called using command line args.
    See docstring at top of file for more info.

    Creates a new fits file with blinded data in it.

    seedstring - a string used to seed pseudo-random parameter shift selection
    initemplate - cosmosis parameter inifile to be used as a template.
    unblinded fits - fits file containing unblinded data
    outfname - if passed, will be name of output file. If not passed, will defaulted to
                [unblinded filename]_BLINDED.fits
      >>> outftag - string added to end of filename to differentiated blinded file
      >>> seedinfname - if true, _seedstring will be added to end of filename
    seedinfits - if true, seedstring will be saved as KEYWORD in fits file. 
                 if false it the string 'notsaved' will be put there instead

    See docstring for run_cosmosis_togen_2ptdict for info and WARNINGS
    about using nz_file and angles_file. 
    """
    #get parameter shifts
    paramshifts = draw_paramshift(seedstring, importfrom = paramshift_module)

    #get blinding factors
    refdict = run_cosmosis_togen_2ptdict(inifile = initemplate, nz_file = nz_file, angles_file = angles_file )
    shiftdict = run_cosmosis_togen_2ptdict(pdict = paramshifts, inifile = initemplate, nz_file = nz_file, angles_file = angles_file )
    factordict = get_factordict(refdict,shiftdict,bftype = bftype)

    #apply them
    #tagstr = bftype+'.'
    if outftag is not None:
        tagstr = outftag # +'_'+tagstr
        if seedinfname:
            tagstr = tagstr+'_'+seedstring
    elif seedinfname:
        tagstr = '_'+seedstring

    if seedinfits:
        storeseed = seedstring
    else:
        storeseed = 'notsaved'
        
    apply2ptblinding_tofits(factordict, origfitsfile = unblindedfits, outfname = outfname, outftag = tagstr, bftype = bftype, storeseed = storeseed)

##############################################################################
##############################################################################
if __name__=="__main__":
    #DEFAULT VALUES
    callscript = False # if true calls script specifically as written below,
                      # otherwise calls according to command line args
                      #  [set to true on command line with --script ]
    unblindedfile = 'pipeline/simulated_nobias_Y3cov.fits'
              #^contains 2pt data to be blinded
              # [set on command line with -u <fname> or --origfits <fname>]

    initemplate = 'pipeline/blinding_params_template.ini'
       #^cosmosis pipeline for generating 2pt dat
       # ref cosmology will be set at whatever its values file is centered at
       # [set on command line with -i <fname> or --ini <fname>]

    bftype = 'add' # type of blinding to use, can be 'add', 'mult', or
        # 'multNOCS' (mult with no cov scaling) >> mult currently disabled

    seed = HARD_CODED_BLINDING #will translate string into some deterministed shift
       #in parameter space
       # [via command line, -s <str> or --seed <str>]

    outfname = None # name of output file containing blinded data.
       #If None, defaults to [unblided fname]_[outftag].fits
       # [via command line, --outfname <str>]
    outftag = '_BLINDED' # string tag that can be used to label output file.
       # By default will be _BLINDED, if a string is passed will be that
       #  [via command line, --outftag <str>]
    paramshift_module = None
    seedinfname = False
    seedinfits = True

    # if we want angles or n(n) from different fits files than are
    #  specified in in initemplate, use these variables WITH CAUTION
    #  They require specific module names that match the DES Y3 2pt pipeline setup
    #  so if that is changed or you're using these for different data, beware.
    #  > it is in general probably safer to not use these and just specify
    #    filenames in the initemplate file
    angles_file = None # file to get angles from.
    nz_file = None

    options,remainder = getopt.getopt(sys.argv[1:],'u:i:s:b:t:o:m:',['origfits=','ini=','seed=','script','outfname=','outftag=','bftype=','paramshiftmodule=','seedinfname=','seedinfits=','anglesfile=','nzfile='])
    #print options
    if ('--script') in options:
        callscript = True
    else:
        for opt, arg in options:
            #print(opt,arg)
            if opt in ('-u','--origfits'):
                unblindedfile = arg
            elif opt in ('-s','--seed'):
                seed = arg 
            elif opt in ('-i', '--ini'):
                initemplate = arg
            elif opt in ('-o','--outfname'):
                outfname = arg
            elif opt in ('-t','--outftag'):
                outftag = arg
            elif opt in ('-b','--bftype'):
                bftype = arg
            elif opt in ('-m','--paramshiftmodule'):
                paramshift_module = arg
            elif opt in ('--seedinfname'):
                seedinfname = arg
            elif opt in ('--seedinfits'):
                seedinfits = arg
            elif opt in ('--nzfile'):
                nz_file = arg
            elif opt in ('--anglesfile'):
                angles_file = arg
                    
    if callscript:
        print("CALLING SCRIPT")
        # for now is just defaults, could set this to something specific later if we wanted
        do2ptblinding(seed,initemplate,unblindedfile,outfname,outftag,bftype,paramshift_module, seedinfname, seedinfits, nz_file=nz_file, angles_file=angles_file)
    else:
        print("LISTENING TO COMMAND LINE ARGS")
        do2ptblinding(seed,initemplate,unblindedfile,outfname,outftag,bftype,paramshift_module, seedinfname, seedinfits, nz_file=nz_file, angles_file=angles_file)
