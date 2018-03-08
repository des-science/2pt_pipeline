
""""

-------------------------------------------------------------------------
This script is an attempt to re-implement some of the blinding code in a way that 
is more flexible (read: less idosyncratic to tests I wrote while trying to learn
the basics of how to work with the 3x2pt pipeline). The goal is to be able
to blind using a cosmosis pipeline setup directly, so without having to pre-compute
a bunch of blinding factors ahead of time. 

The workhorse function that puts everything together is do2ptblinding(). This script 
can be called with command line arguments, or by calling that function in another script.

The script uses cosmosis, so you'll need to  source its setup file before running this.
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

    #TO TEST:
    # same string -> same parameter shifts?
    # calling distribution from another file actualy works
    # ssame string -> same shift when calling another file
    
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
def run_cosmosis_togen_2ptdict(pdict={},inifile='blinding_params_template.ini',outf ='_TEMPOUTPUT_2pt.fits',deleteoutf = True):
    """ 
    Runs cosmosis pipeline to generate 2pt functions. In practice, saves them to a 
    temporary output fits file, which it then reads and then deletes. 

    If a parameter value is passed through the dictionary pdict...
        ...and SHIFTS = False, the param it will be set to that value.
        ...and SHIFTS = True, the param will be shifted by that amount compared to values file
    Otherwise, it will be set to whatever value is in values file.
    
    outf - is name of output fits file containing 2pt data
    deleteoutf - (default True) tells script whether or not to delete the output file. 
               Defaults to True since for blinding we don't want to hang onto the input 2pt files

    If a parameter value is passed through the dictionary pdict, it will be set to that value.
    Otherwise, it will be set to whatever value is in values file.

    Output files will be put in outdir, fbase will be the start of the filenames. 

    Note that you'll need to source the cosmosis setup file before running this.
    """
    # # THINGS TO TEST
    # #  - does this work? YES, it runs
    # #  - does it matter if ini file is on test sampler or multinest?
    # #  - check that blinding factor is applied as expected
    # #  - does this work if we use the cosmosis-standard-library version of save_2pt instead of the 6x2pt version?
    # #     no. The codes are different, but subtley enough that I can't quickly see why.
    # #     will need to fix this if people want to work on other branches of cosmosi-des-library (this is for des_cmp_2pt)
    # #     Can we edit this to not have to save and delete a fits file?
    
    from cosmosis.runtime.config import Inifile
    from cosmosis.runtime.pipeline import LikelihoodPipeline
    from cosmosis.datablock.cosmosis_py import block
    ini=Inifile(inifile)
    print '     cosmosis ini object initialized from',inifile
    #find savesample module to be able to edit filename
    ini.set('save_2pt','filename',outf)
    print 'outfile is set to',ini.get('save_2pt','filename')
    
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

    #read in 2pt data
    print 'looking for',outf,os.path.isfile(outf)
    twoptdict = get_twoptdict_fromfits_1darrays(outf)
    #delete outfile
    if deleteoutf:
        print "REMOVING",outf #test before using
        os.remove(outf)
    return twoptdict

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
            outdict[xkey] = xgrid
            outdict[ykey] = ygrid
                        
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
    ratiodict = {}
    for key in refdict:
        end = key[key.rfind('_')+1:]
        if end in ['ell','l','theta']:
            ratiodict[key] = refdict[key]
        else:
            if bftype=='mult' or bftype=='multNOCS':
                ratiodict[key] = shiftdict[key]/refdict[key]
            elif bftype=='add':
                ratiodict[key] = shiftdict[key] - refdict[key]
            else:
                raise ValueError('In get_factordict: blinding factor type not recognized')
    return ratiodict

#===============================================================
#   HELPER FUNCTIONS FOR WORKIGN WITH DICTS AND FITS FILES
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
def get_data_from_dict_for_2pttype(type1,type2,datadict):
    """ Given strings identifying the type of 2pt data in a fits file 
    and a dictionary of 2pt data (i.e. the blinding factors), 
    returns the data from the dictionary matching those types."""
    #spectra type codes in fits file, under hdutable.header['quant1'] and quant2

    xkey,ykey = get_dictkey_for_2pttype(type1,type2)
    xfromdict = datadict[xkey]
    yfromdict = datadict[ykey]
    return xfromdict,yfromdict

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
    bin1fromfits = table.data['BIN1'] #which bin is quant1 from?
    Nbin1 = np.max(bin1fromfits) #labels start at 1, not 0
    bin2fromfits = table.data['BIN2'] #which bin is quant2 from?
    Nbin2 = np.max(bin2fromfits)
    xfromfits = table.data['ANG']
    xfromdict,yfromdict = get_data_from_dict_for_2pttype(type1,type2,dictdata)
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
    # - does this run?
    # - does filenaming work as expected?
    # - does applied blinding factor match what we expect?
    #    for add, for mult, for multNOCS?
    # - does cov scaling match what we expect?

    
    print 'apply2ptblinding for',origfitsfile
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
            if table.header.get('2PTDATA'):
                if bftype=='mult' or bftype=='multNOCS':
                    table.data['value'] *= get_dictdat_tomatch_fitsdat(table, factordict)
                elif bftype=='add':
                    table.data['value'] += get_dictdat_tomatch_fitsdat(table, factordict)
                else:
                    raise ValueError('bftype {0:s} not recognized'.format(bftype))
                
                #add new header entry to note that blinding has occurred, and what type
                table.header['BLINDED'] = bftype
            elif table.header.get('COVDATA') and bftype=='mult':
                Ndat = table.data.shape[0] #length of datavector

                #get covmat info:
                covmat = covtable.data
                headerkeys = covtable.header.keys()
                names = []
                startinds =[]
                for i in xrange(len(headerkeys)):
                    key = headerkeys[i]
                    if key[:5] == 'STRT_':
                        startinds.append(covtable.header[key])
                    if key[:5] == 'NAME_':
                        name = covtable.header[key]
                        names.append(name)
                startinds.append(Ndat)
                
                bf = np.zeros(Ndat)
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
                [unblinded filename]_[seedstring].fits
    """
    #get parameter shifts
    paramshifts = draw_paramshift(seedstring, importfrom = paramshift_module)

    #get blinding factors
    deleteoutfs = True #would set to false for testing
    refdict = run_cosmosis_togen_2ptdict(inifile = initemplate, outf = '_TEMP_ref_2pt.fits',deleteoutf = deleteoutfs)
    shiftdict = run_cosmosis_togen_2ptdict(pdict = paramshifts, inifile = initemplate, outf = '_TEMP_shifted_2pt.fits',deleteoutf = deleteoutfs)
    factordict = get_factordict(refdict,shiftdict,bftype = bftype)

    #apply them
    if outftag is None: #may set this for testing 
        tagstr = ''
    else:
        tagstr = outftag+'_'
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
        

