
""""
blind_2pt_usingcosmosis.py
-------------------------------------------------------------------------
This script will apply a blinding factor to 2pt functios stored in a fits file, using
cosmosis to compute the blinding factors. 

The workhorse function that puts everything together is do2ptblinding(). This script 
can be called with command line arguments, or by calling that function in another script.

The script uses cosmosis, so you'll need to  source its setup file before running this.

Currently the angle ranges for the angular bins is hard coded in; look for "Passed check: 
blinding factor and fits file have same angle values" to print to make sure you're applying
factors at the right angles.
............................................................
Arguments:
  -u, --origfits : name of unblinded fits file
  -i, --ini : ini file containing template for generating 2pt functions. Should
                -> use a binning that matches that of the origfits file
                -> Reference a values file centered at desired reference cosmology
                -> contain the save_2pt module.
  -s, --seed: string used to seed parameter shift selection
  -t, --bftype: blinding factor type. Can be 'add', 'mult', or 'multNOCS' (mult with no cov scaling)

  -o, --outfname : output filename; only set for testing,  default behavior is
               outfname = [origfits]_[seed].fits
  -t, --outftag : string to label output filename, for testing purposes. If set when
               outfname is None (default), then outfname = [origfits]_[outftag]_[seed].fits
  -m, --paramshiftmodule : string string of a python filename (minus the .py) which can be
                used to import a function to draw parameter shifts (see draw_paramshift for more info)
............................................................

What it does:
1. Using a string seed, pseudo-randomly draw a shift in parameters. This will either
   be drawn from a flat distribution in some predefined parameter ranges, or from 
   a distribution defined in paramshiftmodule. Return dictionary of shifts
   where keys are parameter names matching those expected by run_cosmosis_togen_2ptdict

2. Using a cosmosis parameter ini file template, compute 2pt functions at reference and
   shifted cosmologies by running the cosmosis  test sampler twice. 

3. Collects the 2pt functions into dictionaries of a specific format 
   (see get_twoptdict_fromfits_1darrays). Deletes  new fits files from step 2 (no peeking!). 

3. Take ratio or difference of 2pt datavectors to get blinding factor, in same dictionary format.

4. Apply blinding factor to desired datafile, saving a new fits file with 
   string key in filename. 

-------------------------------------------------------------------------
Script maintained by Jessica Muir (jlmuir@umich.edu).
last update 3/7/18 (NEEDS SIGNIFICANT TESTING)
"""

import numpy as np
import getopt, sys, shutil, os
from astropy.io import fits
import hashlib
import importlib
import time
from scipy.interpolate import interp1d

#############################################################################
# FUNCTIONS FOR GETTING SHIFT IN PARAMETER SPACE
#############################################################################

DEFAULT_PARAM_RANGE = {'cosmological_parameters--sigma8_input':(0.834-3*.04,0.834+3*0.04),\
                       'cosmological_parameters--w':(-1.5,-.5)}

def draw_paramshift(seedstring='blinded', ranges = DEFAULT_PARAM_RANGE,importfrom = None):
    """
    Given a string seed, pseudo-randomly draws shift in parameter space. 

    By default, draws random values for all the parameters named in 
    the list params2shift, from ranges defined in the dictionary ranges. Note:
      - all parameters names in params2shift must also be keys in ranges
      - parameter names should match those used by cosmosis
      - ranges is a dictionary with parameter names as keys, tuples with (min,max)
            values, between which parameter will be drawn from a flat distribution

    importfrom:
    If you'd like to draw from a distribution, define a function for it called
    draw_paramshift in another .py file in the same directory as this script,
    making sure it uses numpy.random in some way to draw a parameter shift.
      - If you pass a string with that file's name as an argument importfrom, 
    that distribution will be used instead of this one. 
      - If this option is used, params2shift and ranges args are ignored. 
      - You must ensure that the dictionary returned has a key 'SHIFTS' associated 
        with a bool to determine how the values in that dictionary are used. 
           pdict['SHIFTS'] = False --> p_shift = pdict['paramname']
           pdict['SHIFTS'] = True  --> p_shift = p_ref + pdict['paramname']

    Make sure any parameters that are shifted here also appear in the 
    key check part of run_cosmosis_togen_2ptdict()
    """
    #do something convoluted to turn the string into a number that can be
    # used to seed numpy.random
    seedind = int(int(hashlib.md5(seedstring).hexdigest(),16)%1.e8)
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
        if 'SHIFTS' not in pdict.keys():
            raise ValueError("Can't find key 'SHIFTS' in pdict. Set to True if the dict contains Delta params, False if it contains param values.")
    
    return pdict

#############################################################################
# FUNCTIONS FOR GETTING BLINDING FACTORS
#############################################################################


# Generate 2pt funcs for set of params
def run_cosmosis_togen_2ptdict(pdict={},inifile='blinding_params_template.ini'):
    """ 
    Runs cosmosis pipeline to generate 2pt functions. 

    If a parameter value is passed through the dictionary pdict...
        ...and SHIFTS = False, the param it will be set to that value.
        ...and SHIFTS = True, the param will be shifted by that amount compared to values file
    Otherwise, it will be set to whatever value is in values file.
    
    If a parameter value is passed through the dictionary pdict, it will be set to that value.
    Otherwise, it will be set to whatever value is in values file.

    Note that you'll need to source the cosmosis setup file before running this.
    """
    # # THINGS TO TEST
    # #  - does it matter if ini file is on test sampler or multinest? No
    # #  - check that blinding factor is applied as expected
    
    from cosmosis.runtime.config import Inifile
    from cosmosis.runtime.pipeline import LikelihoodPipeline
    from cosmosis.datablock.cosmosis_py import block
    ini=Inifile(inifile)
    print '     cosmosis ini object initialized from',inifile
    #ini.set('fits_nz','nz_file',unblfile)
    
    pipeline = LikelihoodPipeline(ini)
    print '     cosmosis pipeline object initialized'
    
    
    doshifts = len(pdict.keys())
    if doshifts:
        SHIFTS = pdict['SHIFTS']
        haveshifted = []
        # set parameters as desired
    for parameter in pipeline.parameters:
        key = str(parameter)
        #print 'working on parameter',key
        
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
            
        # need to set all of the parameters to be fixed for run_parameters([]) to work
        pipeline.set_fixed(parameter.section, parameter.name, parameter.start)

    if doshifts:
        print 'haveshifted',haveshifted
    if doshifts and (len(haveshifted)!= len(pdict.keys()) -1):
        print "WARNING: YOU ASKED FOR SHIFTS IN PARAMTERS NOT IN THE COSMOSIS PIPELINE."
        print "  asked for shifts in:",pdict.keys()
        print "  did shifts in:",haveshifted
    
    # run pipeline to generate 2pt functions for that cosmology
    print "RUNNING PIPELINE=============== [{0:s}]".format(time.strftime("%Y-%m-%d %H:%M"))
    data = pipeline.run_parameters([])
    print ' at end of run_cosmosis_togen_datavec [{0:s}]'.format(time.strftime("%Y-%m-%d %H:%M"))
    twoptdict = get_twoptdict_from_pipelinedata(data)

    return twoptdict

#============================================================
def get_twoptdict_from_pipelinedata(data):
    """
    Given datablock object returned by cosmosis pipeline.run_parameters([]),
    get twopt data in a dictionary whose format matches what you'd get
    if you saved to fits via save_2pt module and read it in with
    get_twoptdict_fromfits_1darrays
    """
    # this theta handling is pulled from save_2pt.py
    # hard coding this isn't ideal, but it'll work for now
    theta_min = 2.5
    theta_max = 250.
    n_theta = 20
    theta = np.logspace(np.log10(theta_min), np.log10(theta_max), n_theta+1, endpoint=True)
    a = theta[:-1]
    b = theta[1:]
    theta = 2./3. * (b**3-a**3)/(b**2-a**2)
    # ^ see Krause et al. methods paper for where this comes from
    #print "Computing factors for theta=",theta

    #these are the types
    galaxy_position_fourier = "GPF"
    galaxy_shear_emode_fourier = "GEF"
    galaxy_shear_bmode_fourier = "GBF"
    galaxy_position_real = "GPR"
    galaxy_shear_plus_real = "G+R"
    galaxy_shear_minus_real = "G-R"

    #we don't apply any bin or angular cuts here. When we apply blinding factors
    # we'll make sure bin numbers and angular bin numbers match up with those in the
    # unblinded fits file. So, as long as the angular and z bin indices correpond to the
    # same angles and bins, cuts will be handled in that stage.
    
    outdict = {}
    if data.has_section('galaxy_xi'):
        types = (galaxy_position_real,galaxy_position_real)
        xkey,ykey = get_dictkey_for_2pttype(types[0],types[1])
        x,y,bins,angbins = spectrum_array_from_block(data,'galaxy_xi',types,theta,True)
        outdict[xkey]=x
        outdict[ykey]=y
        outdict[ykey+'_bins'] = bins
        outdict[ykey+'_angbins'] = angbins
    if data.has_section('galaxy_shear_xi'):
        types = (galaxy_position_real,galaxy_shear_plus_real)
        xkey,ykey = get_dictkey_for_2pttype(types[0],types[1])
        x,y,bins,angbins = spectrum_array_from_block(data,'galaxy_shear_xi',types,theta,True)
        outdict[xkey]=x
        outdict[ykey]=y
        outdict[ykey+'_bins'] = bins
        outdict[ykey+'_angbins'] = angbins
    if data.has_section('shear_xi'):
        #xip
        types = (galaxy_shear_plus_real,galaxy_shear_plus_real)
        xkey,ykey = get_dictkey_for_2pttype(types[0],types[1])
        x,y,bins,angbins = spectrum_array_from_block(data,'shear_xi',types,theta,True,bin_format = 'xiplus_{0}_{1}')
        outdict[xkey]=x
        outdict[ykey]=y
        outdict[ykey+'_bins'] = bins
        outdict[ykey+'_angbins'] = angbins
        #xim
        types = (galaxy_shear_minus_real,galaxy_shear_minus_real)
        xkey,ykey = get_dictkey_for_2pttype(types[0],types[1])
        x,y,bins,angbins = spectrum_array_from_block(data,'shear_xi',types,theta,True,bin_format = 'ximinus_{0}_{1}')
        outdict[xkey]=x
        outdict[ykey]=y
        outdict[ykey+'_bins'] = bins
        outdict[ykey+'_angbins'] = angbins
            
    return outdict

#-----------------------------------------------            
def spectrum_array_from_block(block, section_name, types,  angle_sample, real_space, bin_format = 'bin_{0}_{1}'):
    """
    Adapting this from save_2pt.py's spectrum_measurement_from_block

    No scale cutting implemented. I think that this is ok, since the fits files 
    currently contain data inside cut regions. Just need to cut out
    bins that aren't in the arrays for the fits files. 

    (as these 2pt dictionaries are set up, we don't track bin info, so
     have to make sure they're in the same order.)
    """

    # for cross correlations we must save bin_ji as well as bin_ij.
    # but not for auto-correlations. Also the numbers of bins can be different
    is_auto = (types[0] == types[1])
    if block.has_value(section_name, "nbin"):
        nbin_a = block[section_name, "nbin"]
    else:
        nbin_a = block[section_name, "nbin_a"]
        nbin_b = block[section_name, "nbin_b"]


    #This is the ell values that have been calculated by cosmosis, not to
    #be confused with the ell values at which we want to save the results
    #(which is ell_sample)
    if real_space:
        # This is in radians
        theory_angle = block[section_name, "theta"]
        angle_sample_radians = np.radians(angle_sample/60.)
    else:
        theory_angle = block[section_name, "ell"]
        angle_sample_radians = angle_sample

    #This is the length of the sample values
    n_angle_sample = len(angle_sample)
    
    #The fits format stores all the measurements
    #as one long vector.  So we build that up here from the various
    #bins that we will load in.  These are the different columns
    value = []
    angle = []
    bin1 = []
    bin2 = []
    angular_bin = []
        
    #Bin pairs. Varies depending on auto-correlation
    for i in xrange(nbin_a):
        if is_auto:
            jmax = i+1
        else:
            jmax = nbin_b
        for j in xrange(jmax):
            #if ((i,j) in bincuts) or (is_auto and (j,i) in bincuts):
            #    print 'cutting out bin combo',i,j,' for ',section_name
            #    continue
                
            #Load and interpolate from the block
            cl = block[section_name, bin_format.format(i+1,j+1)]
            #Convert arcmin to radians for the interpolation
            cl_interp = SpectrumInterp(theory_angle, cl)
            cl_sample = cl_interp(angle_sample_radians)
            bin1.append(np.repeat(i + 1, n_angle_sample))
            bin2.append(np.repeat(j + 1, n_angle_sample))
            angular_bin.append(np.arange(n_angle_sample))
            #Build up on the various vectors that we need
            value.append(cl_sample)
            angle.append(angle_sample)

    #Convert all the lists of vectors into long single vectors
    value = np.concatenate(value)
    angle = np.concatenate(angle)
    bin1 = np.concatenate(bin1)
    bin2 = np.concatenate(bin2)
    bins = (bin1,bin2)
    angularbin = np.concatenate(angular_bin)

    return angle, value, bins, angularbin

#-----------------------------------------------        
class SpectrumInterp(object):
    """
    This is copied from 2pt_like, for use in spectrum_array_from_block
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
def get_twoptdict_fromfits_1darrays(fitsfile):
    """
    Given  fits files containing two point functions, read in data.

    Returns a dictionary with keys specified in get_2pttype_fordictkey
    with angle and 2pt data in entries specified by xkey and ykey.
    """
    outdict = {}
    # for indexing: [which 2pt func][which input file]
    # for each file there will be a dictionary where tuples (bin1,bin2) are keys
    h = fits.open(fitsfile)
    for table in h:
        if table.header.get('2PTDATA'):
            t = table
            type1 = t.header['QUANT1']
            type2 = t.header['QUANT2']
            xkey,ykey = get_dictkey_for_2pttype(type1,type2)
            xgrid = t.data['ANG']
            ygrid = t.data['VALUE']
            bin1 = t.data['BIN1']
            bin2 = t.data['BIN2']
            angbins = t.data['ANGBIN']
            outdict[xkey] = xgrid
            outdict[ykey] = ygrid
            outdict[ykey+'_bins'] = (bin1,bin2)
            outdict[ykey+'_angbins'] = angbins
    h.close()
    return outdict 
#============================================================
def get_factordict(refdict,shiftdict,bftype='add'):
    """
    Given two point dictionaries for reference and shifted cosmology,
    returns dictionary of blinding factors. 

    bftype = what kind of blinding factor is it? 
             'add' : bf = - ref + shift
             'mult': bf = shift/ref
    """
    #print bftype,'in get_factordict'
    factordict = {}
    for key in refdict:
        end = key[key.rfind('_')+1:]
        if end in ['ell','l','theta','bins','angbins']:
            factordict[key] = refdict[key]
        else:
            if bftype=='mult' or bftype=='multNOCS':
                #print 'dividing!'
                factordict[key] = shiftdict[key]/refdict[key]
            elif bftype=='add':
                #print 'adding'
                factordict[key] = shiftdict[key] - refdict[key]
            else:
                raise ValueError('In get_factordict: blinding factor type not recognized')
    return factordict

#===============================================================
#   HELPER FUNCTIONS FOR WORKING WITH DICTS AND FITS FILES
#  (Expects shear and number density data with these keys:)
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

def get_dictkey_forcovname(covname):
    """
    Convert string used to label covmat entries into dictkeys that can be used
    with dictionaries read in with get_twoptdict_fromfits_1darrays()
    """
    if covname=='xip':
        ykey = 'shear_shear_xip'
        xkey = 'shear_shear_theta'
    elif covname=='xim':
        ykey = 'shear_shear_xim'
        xkey =  'shear_shear_theta'
    elif covname=='gammat':
        ykey = 'gal_shear_xi'
        xkey = 'gal_shear_theta'
    elif covname=='wtheta':
        ykey = 'gal_gal_xi'
        xkey = 'gal_gal_theta'
    else:
        raise ValueError("Spectra type {0:s} not recognized in get_dictkey_forcovname.".format(covname))
    return xkey,ykey
#---------------------------------------     
def get_dictkey_for_2pttype(type1,type2):
        #spectra type codes in fits file, under hdutable.header['quant1'] and quant2
    galaxy_position_fourier = "GPF"
    galaxy_shear_emode_fourier = "GEF"
    galaxy_shear_bmode_fourier = "GBF"
    galaxy_position_real = "GPR"
    galaxy_shear_plus_real = "G+R"
    galaxy_shear_minus_real = "G-R"
    
    if type1==galaxy_position_fourier and type2 == galaxy_position_fourier:
        ykey = 'gal_gal_cl'
        xkey = 'gal_gal_l'
    elif (type2==galaxy_shear_emode_fourier and type1 == galaxy_position_fourier):
        ykey ='gal_shear_cl'
        xkey = 'gal_shear_l'
    elif (type1==galaxy_shear_emode_fourier and type2 == galaxy_shear_emode_fourier):
        ykey = 'shear_shear_cl'
        xkey = 'shear_shear_l'
    elif type1==galaxy_position_real and type2 == galaxy_position_real:
        ykey = 'gal_gal_xi'
        xkey = 'gal_gal_theta'
    elif (type2==galaxy_shear_plus_real and type1 == galaxy_position_real):
        # not symmetric
        ykey = 'gal_shear_xi'
        xkey = 'gal_shear_theta'
    elif (type1==galaxy_shear_plus_real and type2 == galaxy_shear_plus_real):
        ykey = 'shear_shear_xip'
        xkey = 'shear_shear_theta'
    elif (type1==galaxy_shear_minus_real and type2 == galaxy_shear_minus_real):
        ykey = 'shear_shear_xim'
        xkey =  'shear_shear_theta'
    else:
        raise ValueError("Spectra type {0:s} - {1:s} not recognized.".format(type1,type2))
    return xkey,ykey
#---------------------------------------     
def get_data_from_dict_for_2pttype(type1,type2,bin1fits,bin2fits,angbfits,datadict):
    """ Given strings identifying the type of 2pt data in a fits file 
    and a dictionary of 2pt data (i.e. the blinding factors), 
    returns the data from the dictionary matching those types."""
    #spectra type codes in fits file, under hdutable.header['quant1'] and quant2

    xkey,ykey = get_dictkey_for_2pttype(type1,type2)
    xfromdict = datadict[xkey]
    yfromdict = datadict[ykey]
    binsdict = datadict[ykey+'_bins']
    b1dict = binsdict[0]
    b2dict = binsdict[1]
    angbdict = datadict[ykey+'_angbins']

    Nentries = bin1fits.size
    xout = np.zeros(Nentries)
    yout = np.zeros(Nentries)
    for i in xrange(Nentries):
        b1 = bin1fits[i]
        b2 = bin2fits[i]
        ab = angbfits[i]
        whichind = (b1==b1dict)*(b2==b2dict)*(ab==angbdict)
        xout[i] = xfromdict[whichind]
        yout[i] = yfromdict[whichind]

    #we want to return a subset of the 'fromdict' arrays
    # for which the angular and z bins match the fits versions
    return xout,yout

#---------------------------------------
def get_dictdat_tomatch_fitsdat(table,dictdata):
    """
    Given table of type fits.hdu.table.BinTableHDU containing 2pt data, 
    and retrieves corresponding data from dictionary (blinding factors).

    # ASSUMES binning and array order in dict is the same as in fits file.
    """
    if not table.header.get('2PTDATA'):
        print "Can't match dict data: this fits table doesn't contain 2pt data. Is named:",table.name
        return
    type1 = table.header['QUANT1']
    type2 = table.header['QUANT2']

    bin1 = table.data['BIN1'] #which bin is quant1 from?
    bin2 = table.data['BIN2'] #which bin is quant2 from?
    angbin = table.data['ANGBIN'] #which angular bin is data from
    xfromfits = table.data['ANG']
    xfromdict,yfromdict = get_data_from_dict_for_2pttype(type1,type2,bin1,bin2,angbin,dictdata)

    if np.all(xfromdict==xfromfits):
        print "Passed check: blinding factor and fits file have same angle values."
    else:
        print "FAILED CHECK: blinding factor and fits file DO NOT have same angle values."
        print '        fits file angles:',xfromfits
        print '  blinding factor angles:',xfromdict
        
    return yfromdict

#############################################################################
# APPLY BLINDING FACTORS TO FITS FILE
#############################################################################

def apply2ptblinding_tofits(factordict, origfitsfile = 'two_pt_cov.fits', outfname = None, outftag = None, justfname = False,bftype='add'):
    """ 
    Given the dictionary of one set of blinding factors, 
    the name of a  of a fits file containing unblinded 2pt data, 
    and (optional) desired output file name or tag, 
    multiplies 2pt data in original fits file by blinding factors and 
    saves results (blinded data) into a new fits file. 

    If argument is passed for outfname, that will be the name of output file,
    if not, will be <input filename>_<outftag>.fits. 

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
    # TO TEST:
    # - does applied blinding factor match what we expect?
    #    for add, for mult, for multNOCS?; multNOCS!=mult, which is bad
    # - does cov scaling match what we expect?
    
    print 'apply2ptblinding for',origfitsfile
    print 'bftype',bftype
    # check whether data is already blinded and whether Nbins match
    for table in fits.open(origfitsfile): #look through tables to find 2ptdata
        if table.header.get('2PTDATA'): 
            if table.header.get('BLINDED'): #check for blinding
                #if entry not there, or storing False -> not already blinded
                raise ValueError('Data is already blinded!')
                return
            
    # set up output file 
    if outfname == None or outfname==origfitsfile: #make sure you can't accidentaly overwrite the original
        #if output filename isn't given, add outftag onto the name of the unblinded file
        if outftag == None or outftag =='': 
            outftag = 'BLINDED-{0:s}-defaulttag'.format(bftype)
        outfname = origfitsfile.replace('.fits','_{0:s}.fits'.format(outftag))

    if not justfname:
        shutil.copyfile(origfitsfile,outfname)

        hdulist = fits.open(outfname,mode='update') #update lets us write over

        # apply blinding factors 
        for table in hdulist: #look all tables
            #workign here uncomment
            if table.header.get('2PTDATA'):
                if bftype=='mult' or bftype=='multNOCS':
                    print 'dividing!'
                    table.data['value'] *= get_dictdat_tomatch_fitsdat(table, factordict)
                elif bftype=='add':
                    print 'adding!'
                    table.data['value'] += get_dictdat_tomatch_fitsdat(table, factordict)
                else:
                    raise ValueError('bftype {0:s} not recognized'.format(bftype))
                
                #add new header entry to note that blinding has occurred, and what type
                table.header['BLINDED'] = bftype
            elif table.header.get('COVDATA') and bftype=='mult':
                Ndat = table.data.shape[0] #length of datavector

                #get covmat info:
                covmat = table.data
                headerkeys = table.header.keys()
                names = []
                startinds =[]
                for i in xrange(len(headerkeys)):
                    key = headerkeys[i]
                    if key[:5] == 'STRT_':
                        startinds.append(table.header[key])
                    if key[:5] == 'NAME_':
                        name = table.header[key]
                        names.append(name)
                startinds.append(Ndat)
                
                bf = np.zeros(Ndat)
                print names,startinds
                for i,covname in enumerate(names):
                    starti = startinds[i]
                    endi = startinds[i+1]
                    xkey,ykey = get_dictkey_forcovname(covname)
                    bf[starti:endi] = factordict[ykey]
                # Do cov_new[i,j] = cov[i,j]*bf[i]*bf[j]     
                table.data = bf.reshhape((bf.size,1))*table.data*bf
            #else:
            #    #print '-------\n',table.name,'\n-------\nNOT TWO POINT DATA'
            #    pass
        
        hdulist.close() # will save new data to file if 'update' was passed when opened
        print ">>>>Stored blinded data in",outfname
    return outfname

#############################################################################
# WRAPPER: Put everything together
#############################################################################
def do2ptblinding(seedstring,initemplate,unblindedfits, outfname = None, outftag = None,bftype='add',paramshift_module = None):
    """
    This is the function that gets called using command line args. 

    Creates a new fits file with blinded data in it.  

    seedstring - a string used to seed pseudo-random parameter shift selection
    initemplate - cosmosis parameter inifile to be used as a template. Should use save_2pt 
         and test modules, and in test module options set save_dir= to nothing
    unblinded fits - fits file containing unblinded data
    outfname - if passed, will be name of output file. If not passed, will defaulted to
                [unblinded filename].[seedstring].fits
    """
    #get parameter shifts
    paramshifts = draw_paramshift(seedstring, importfrom = paramshift_module)

    #get blinding factors
    refdict = run_cosmosis_togen_2ptdict(inifile = initemplate)
    shiftdict = run_cosmosis_togen_2ptdict(pdict = paramshifts, inifile = initemplate)
    factordict = get_factordict(refdict,shiftdict,bftype = bftype)

    #apply them
    tagstr = bftype+'.'
    if outftag is not None:
        tagstr = outftag+'_'+tagstr
    apply2ptblinding_tofits(factordict, origfitsfile = unblindedfits, outfname = outfname, outftag = tagstr+seedstring, bftype = bftype)
    
##############################################################################
##############################################################################
if __name__=="__main__":
    #DEFAULT VALUES
    callscript = False # if true calls script specifically as written below, 
                      # otherwise calls according to command line args
                      #  [set to true on command line with --script ]
    unblindedfile = 'simulated_nobias_Y3cov.fits' #contains 2pt data to be blinded
                      # [set on command line with -u <fname> or --origfits <fname>]

    initemplate = 'blinding_params_template.ini' #cosmosis pipeline for generating 2pt dat (needes save_2pt module!)
    # ref cosmology will be set at whatever its values file is centered at
           # [set on command line with -i <fname> or --ini <fname>]

    bftype = 'add' # type of blinding to use, can be 'add', 'mult', or 'multNOCS' (mult with no cov scaling)
    
    seed = 'blinded' #will translate string into some deterministed shift in parameter space
                      # [via command line, -s <str> or --seed <str>]

    outfname = None # name of output file containing blinded data. If None, defaults to [unblided fname]_[seed].fits
                      # [via command line, --outfname <str>]
    outftag = None # string tag that can be used to label output file. If not None, but outfname is None,
                    # will give outnames like [unblided fname]_[outftag]_[seed].fits
                      # [via command line, --outftag <str>]
    paramshift_module = None 
                      

    options,remainder = getopt.getopt(sys.argv[1:],'u:i:s:b:t:o:m:',['origfits=','ini=','seed=','script','outfname=','outftag=','bftype=','paramshiftmodule='])
    #print options
    if ('--script') in options:
        callscript = True
    else:
        for opt, arg in options:
            print opt
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

    if callscript:
        print "CALLING SCRIPT"
        # for now is just defaults, could set this to something specific later if we wanted
        do2ptblinding(seed,initemplate,unblindedfile,outfname,outftag,bftype,paramshift_module)
    else:
        print "LISTENING TO COMMAND LINE ARGS"
        do2ptblinding(seed,initemplate,unblindedfile,outfname,outftag,bftype,paramshift_module)
        

