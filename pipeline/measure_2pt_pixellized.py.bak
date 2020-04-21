from __future__ import print_function, division
import numpy as np
import treecorr
import twopoint
import h5py
import fitsio as fio
import healpy as hp
from numpy.lib.recfunctions import append_fields, rename_fields
from .stage import PipelineStage, TWO_POINT_NAMES
import os
import sys
import yaml
import destest
import importlib
import glob
import timeit

import pickle
import kmeans_radec
from scipy import spatial
from kmeans_radec import KMeans, kmeans_sample

global_measure_2_point = None

destest_dict_ = {
    'output_exists' : True,
    'use_mpi'       : False,
    'source'        : 'hdf5',
    'dg'            : 0.01
    }

def create_destest_yaml( params, name, cal_type, group, table, select_path, name_dict ):
    """
    Creates the input dictionary structure from a passed dictionary rather than reading froma yaml file.
    """

    destest_dict = destest_dict_.copy()
    destest_dict['load_cache'] = params['load_cache']
    destest_dict['name'] = name
    destest_dict['output'] = params['output']
    destest_dict['filename'] = params['datafile']
    destest_dict['param_file'] = params['param_file']
    destest_dict['cal_type'] = cal_type
    destest_dict['group'] = group
    destest_dict['table'] = table
    destest_dict['select_path'] = select_path
    destest_dict['e'] = [name_dict.shape_dict['e1'],name_dict.shape_dict['e2']]
    destest_dict['Rg'] = [name_dict.shape_dict['m1'],name_dict.shape_dict['m2']]

    return destest_dict

def load_catalog(pipe_params, name, cal_type, group, table, select_path, name_dict, inherit=None, return_calibrator=None):
    """
    Loads data access and calibration classes from destest for a given yaml setup file.
    """
    # Input yaml file defining catalog
    params = create_destest_yaml(pipe_params, name, cal_type, group, table, select_path, name_dict)
    # Load destest source class to manage access to file
    source = destest.H5Source(params)
    # Load destest selector class to manage access to data in a structured way
    if inherit is None:
        sel = destest.Selector(params,source)
    else:
        sel = destest.Selector(params,source,inherit=inherit)
    # Load destest calibrator class to manage calibration of the catalog
    if return_calibrator is not None:
        cal = return_calibrator(params,sel)
        return sel, cal
    else:
        return sel

def task_full(ijk):

    def save_obj1(name, obj):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    i,j,[o1,o2,jc],k = ijk 
    
    if 1==1:
        path_save = global_measure_2_point.params['run_directory']+'/2pt/{0}_{1}_{2}/'.format(i,j,k)
        if not os.path.exists(path_save):
            os.mkdir(path_save)
        if not os.path.exists(path_save+'_full.pkl'):
            xx = global_measure_2_point.calc_correlation(i,j,k, full = True)
            
            if k ==2:
                ndd=(np.sum(global_measure_2_point.Npairs['{0}_{1}'.format(i,j)]['jck_N'][:,0]))*(np.sum(global_measure_2_point.Npairs['{0}_{1}'.format(i,j)]['jck_N'][:,1]))
                ndr=(np.sum(global_measure_2_point.Npairs['{0}_{1}'.format(i,j)]['jck_N'][:,0]))*(np.sum(global_measure_2_point.Npairs['{0}_{1}'.format(i,j)]['jck_N'][:,3]))
                nrd=(np.sum(global_measure_2_point.Npairs['{0}_{1}'.format(i,j)]['jck_N'][:,2]))*(np.sum(global_measure_2_point.Npairs['{0}_{1}'.format(i,j)]['jck_N'][:,1]))
                nrr=(np.sum(global_measure_2_point.Npairs['{0}_{1}'.format(i,j)]['jck_N'][:,2]))*(np.sum(global_measure_2_point.Npairs['{0}_{1}'.format(i,j)]['jck_N'][:,3]))
                norm=[1.,ndd/ndr,ndd/nrd,ndd/nrr]
                for ei in range(4):
                    xx[ei] = xx[ei]*norm[ei]
            save_obj1(path_save+'_full',xx)


            
def task(ijk):

    def save_obj1(name, obj):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    i,j,[o1,o2,jc],k = ijk 
    
    if k ==0:
        type_corr = 'shear_shear'
    if k ==1:
        type_corr = 'shear_pos'
    if k ==2:
        type_corr = 'pos_pos'
    
    path_save = global_measure_2_point.params['run_directory']+'/2pt/{0}_{1}_{2}/'.format(i,j,k)
    if not os.path.exists(path_save):
        os.mkdir(path_save)
        
    lla = '{0}'.format(jc)
    if not os.path.exists(path_save+lla+'.pkl'):
        #print "computing ", path_save+lla

        pairsCC1 = global_measure_2_point.calc_correlation(i,j,k,o1, [jc])
        pairsCC2 = global_measure_2_point.calc_correlation(i,j,k,[jc], o2)
        pairs_auto = global_measure_2_point.calc_correlation(i,j,k,[jc], [jc])

        dict_m = dict()
        dict_m.update({'c1':pairsCC1})
        dict_m.update({'c2':pairsCC2})
        dict_m.update({'a':pairs_auto})
        save_obj1(path_save+lla,dict_m)

def save_obj(name, obj):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)#, encoding='latin1')
def covariance_jck(TOTAL_PHI,jk_r,type_cov):
      if type_cov=='jackknife':
          fact=(jk_r-1.)/(jk_r)

      elif type_cov=='bootstrap':
          fact=1./(jk_r)
      #  Covariance estimation

      average=np.zeros(TOTAL_PHI.shape[0])
      cov_jck=np.zeros((TOTAL_PHI.shape[0],TOTAL_PHI.shape[0]))
      err_jck=np.zeros(TOTAL_PHI.shape[0])


      for kk in range(jk_r):
        average+=TOTAL_PHI[:,kk]
      average=average/(jk_r)

     # print average
      for ii in range(TOTAL_PHI.shape[0]):
         for jj in range(ii+1):
              for kk in range(jk_r):
                cov_jck[jj,ii]+=TOTAL_PHI[ii,kk]*TOTAL_PHI[jj,kk]

              cov_jck[jj,ii]=(-average[ii]*average[jj]*jk_r+cov_jck[jj,ii])*fact
              cov_jck[ii,jj]=cov_jck[jj,ii]

      for ii in range(TOTAL_PHI.shape[0]):
       err_jck[ii]=np.sqrt(cov_jck[ii,ii])
     # print err_jck

      #compute correlation
      corr=np.zeros((TOTAL_PHI.shape[0],TOTAL_PHI.shape[0]))
      for i in range(TOTAL_PHI.shape[0]):
          for j in range(TOTAL_PHI.shape[0]):
            corr[i,j]=cov_jck[i,j]/(np.sqrt(cov_jck[i,i]*cov_jck[j,j]))

      average=average*fact
      return {'cov' : cov_jck,
              'err' : err_jck,
              'corr':corr,
              'mean':average}
    
    
class Measure2Point(PipelineStage):
    name = "2pt"
    inputs = {
        "weight"        : ("nofz", "weight.npy")          ,
        "nz_source"     : ("nofz", "nz_source_zbin.h5")  ,
        "nz_lens"       : ("nofz", "nz_lens_zbin.npy")    ,
        "randoms"       : ("nofz", "randoms_zbin.npy")    ,
        "gold_idx"      : ("nofz", "gold_idx.npy")        ,
        "lens_idx"      : ("nofz", "lens_idx.npy")        ,
        "ran_idx"       : ("nofz", "ran_idx.npy")         ,
        "nofz_meta"     : ("nofz", "metadata.yaml")       ,
    }
    outputs = {
        "xip"    : "{rank}_xip.txt",
        "xim"    : "{rank}_xim.txt",
        "gammat" : "{rank}_gammat.txt",
        "wtheta" : "{rank}_wtheta.txt",
        "any"    : "*{rank}*",
    }

    def __init__(self, param_file):
        """
        Initialise object and load catalogs.
        """
        super(Measure2Point,self).__init__(param_file)

        # Load metadata (needed for mean shear)
        self.load_metadata()

        # Default value for random subsampling factor
        if 'ran_factor' not in self.params:
            self.params['ran_factor'] = 999

        # A dictionary to homogenize names of columns in the hdf5 master catalog 
        self.Dict = importlib.import_module('.'+self.params['dict_file'],'pipeline')
        print ('using dictionary: ',self.params['dict_file'])
           
        # Load data and calibration classes
        if self.params['has_sheared']:
            self.source_selector, self.source_calibrator = load_catalog(
                self.params, 'mcal', 'mcal', self.params['source_group'], self.params['source_table'], self.params['source_path'], self.Dict, return_calibrator=destest.MetaCalib)
            self.lens_selector, self.lens_calibrator = load_catalog(
                self.params, 'lens', None, self.params['lens_group'], self.params['lens_table'], self.params['lens_path'], self.Dict, return_calibrator=destest.NoCalib)
            self.gold_selector = load_catalog(
                self.params, 'gold', 'mcal', self.params['gold_group'], self.params['gold_table'], self.params['gold_path'], self.Dict, inherit=self.source_selector)
            self.pz_selector = load_catalog(
                self.params, 'pz', 'mcal', self.params['pz_group'], self.params['pz_table'], self.params['pz_path'], self.Dict, inherit=self.source_selector)
            self.ran_selector = load_catalog(
                self.params, 'ran', None, self.params['ran_group'], self.params['ran_table'], self.params['ran_path'], self.Dict)
        else:
            self.source_selector, self.source_calibrator = load_catalog(
                self.params, 'mcal', 'mcal', self.params['source_group'], self.params['source_table'], self.params['source_path'], self.Dict, return_calibrator=destest.NoCalib)
            self.lens_selector, self.lens_calibrator = load_catalog(
                self.params, 'lens', None, self.params['lens_group'], self.params['lens_table'], self.params['lens_path'], self.Dict, return_calibrator=destest.NoCalib)
            self.gold_selector = load_catalog(
                self.params, 'gold', 'mcal', self.params['gold_group'], self.params['gold_table'], self.params['gold_path'], self.Dict, inherit=self.source_selector)
            self.pz_selector = load_catalog(
                self.params, 'pz', 'mcal', self.params['pz_group'], self.params['pz_table'], self.params['pz_path'], self.Dict, inherit=self.source_selector)
            self.ran_selector = load_catalog(
                self.params, 'ran', None, self.params['ran_group'], self.params['ran_table'], self.params['ran_path'], self.Dict)

        
        # Added!
        self.source_regions_selector = load_catalog(self.params, 'reg_mcal', 'mcal', self.params['gold_regions_group'], self.params['gold_regions_table'], self.params['gold_regions_path'], self.Dict, inherit=self.source_selector)
        self.lens_regions_selector = load_catalog(self.params, 'reg_lens', 'mcal', self.params['lens_regions_group'], self.params['lens_regions_table'], self.params['lens_regions_path'], self.Dict,inherit=self.source_selector)
        self.lens_random_regions_selector = load_catalog(self.params, 'reg_ran', None, self.params['lens_randoms_regions_group'], self.params['lens_randoms_regions_table'], self.params['lens_randoms_regions_path'], self.Dict)

        
        
        self.Dict.ind = self.Dict.index_dict #a dictionary that takes unsheared,sheared_1p/1m/2p/2m as u-1-2-3-4 to deal with tuples of values returned by get_col()
        
        self.setup_jaccknife()
        global global_measure_2_point
        global_measure_2_point = self

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


    def setup_jaccknife(self):
        cals =[]
        if self.params['lens_group'] != 'None':
            cals.append(self.lens_calibrator)
        if self.params['source_group'] != 'None':
            cals.append(self.source_calibrator)
            
        jack_info_tot = dict()
        for cal in cals:
            if (type(cal)==destest.NoCalib) and (cal.params['cal_type'] is None):
                nbin=self.lens_zbins
            else:
                nbin=self.zbins

            for i in range(nbin):
                
                jack_info = dict()
                if type(cal)==destest.NoCalib and (cal.params['cal_type'] is None):
                    R1,R2,mask,w,rmask = self.get_zbins_R(i,cal)
                    region_lenses =   self.lens_regions_selector.source.read('region')[self.Dict.ind['u']][cal.selector.get_mask()[self.Dict.ind['u']]][mask]

                    jack_info.update({'hpix' : region_lenses})

                    region_lenses_randoms  = (self.lens_random_regions_selector.get_col(self.Dict.regions_dict['region'])[self.Dict.ind['u']][rmask])
                    
                    jack_info.update({'hpix_randoms' : region_lenses_randoms})
                    jack_info_tot.update({'{0}_lens'.format(i):jack_info})
                else:
                    R1,R2,mask,w = self.get_zbins_R(i,cal)
                    region_sources =self.source_regions_selector.get_col(self.Dict.regions_dict['region'])[self.Dict.ind['u']][mask]
                    #jack_info.update({'n_jck' :  self.jack_dict_tot['n_jck']})                                   
                    jack_info.update({'hpix' : region_sources})
                    jack_info_tot.update({'{0}_shears'.format(i):jack_info})
                    #jack_info.update({'hpix_jackknife' : hpix})
                    #jack_info.update({'ra' : ra_mask})
                    #jack_info.update({'dec' : dec_mask})
                    #jack_info.update({'centers' : centers})
                    #jack_info.update({'centers_tree' : centers_tree})
                                
        # compute distance between centers ******
        try:
            f = h5py.File(self.lens_calibrator.params['filename'],'r')
        except:
            f = h5py.File(self.source_calibrator.params['filename'],'r')
        ra_regions = f['regions']['centers']['ra']
        dec_regions = f['regions']['centers']['dec']
        centers =np.array(zip(ra_regions,dec_regions))
        jack_info_tot.update({'max_distance':np.array(f['regions']['centers']['dist'])})
        jack_info_tot.update({'n_jck':np.array(f['regions']['centers']['number'][0])})
        jack_info_tot.update({'centers':centers})
        self.jack_dict_tot = jack_info_tot
    def setup_jobs(self,pool):
        """
        Set up the list of jobs that must be distributed across nodes.
        """

        def distance(cnt,FACT_DIST = 2):

            """Finds the minimum distance to a center for each center (cnt)
               Fixes double of this distance (fact_dist) as the criteria for not considering correlations,
               which is a conservative choice. This distance has to be at least 4 times the
               maximum angular separation considered. Centers beyond this distance will not be
               considered in the correlations.
            """

            # Find the minimum distance to a center for each center.
            dist = np.array([np.sort([dist_cent(cnt[i],cnt[j]) for i in range(len(cnt))])[1] for j in range(len(cnt))])
            dist = (dist)*FACT_DIST


            max_sep = self.params['tbounds'][1] * 1./60. #ASSUMES INPUT IN ARCMIN!

            # Check that the distance is at least 4 times the maximum angular separation.
            return np.array( [ 4.*max_sep if x < 4.*max_sep else x for x in dist] )


        def dist_cent(a, b):
            """Angular distance between two centers (units: degrees). Makes use of spherical law of cosines.
            """
            todeg = np.pi/180.
            a = a* todeg
            b = b* todeg
            cos = np.sin(a[1])*np.sin(b[1]) + np.cos(a[1])*np.cos(b[1])*np.cos(a[0]-b[0])
            return np.arccos(cos)/( todeg)

    
        def load_max_distance_region(distance_file):

            def load_obj(name):
                with open(name + '.pkl', 'rb') as f:
                    return pickle.load(f)#, encoding='latin1')

            return load_obj(distance_file)
            
        
        
        
        # Get max number of tomographic bins between lenses and sources
        if self.params['lens_group'] != 'None':
            nbin=max(self.lens_zbins,self.zbins)
        else:
            nbin=self.zbins

        print ('setting up ',nbin,'tomographic bins')
        

        cnt = self.jack_dict_tot['centers']
        
        #compute distances between centers
        print ('compute distance between centers')
        center_min_dis = distance(cnt)
        

        max_dist_region = self.jack_dict_tot['max_distance']        
        max_sep = self.params['tbounds'][1] * 1./60. #ASSUMES INPUT IN ARCMIN!
        
        # define wich jackknife region pairs need to be included in the correlation computation
        print ("jackknife pairs selection")
        a=np.concatenate((np.array([[(i,j) for i in range(self.jack_dict_tot['n_jck'])] for j in range(self.jack_dict_tot['n_jck'])])))
        
        sel = np.array([max([0.,(dist_cent(cnt[i],cnt[j]) - (max_dist_region[i]+max_dist_region[j]))]) < (self.params['separation_factor'] * max_sep )for (i,j) in a])
        b = a[sel]  
        b1 = []

        

        
        # this is to set up the jackknife pairs when correlation functions are computed
        for jc in (np.unique(b[:, 1])):
            mask = (b[:, 1] == jc) & (b[:, 0] != jc)
            othersample = b[mask, 0]
            mask = (b[:, 0] == jc) & (b[:, 1] != jc)
            othersample1 = b[mask, 1]
            b1.append([othersample,othersample1,jc])
        

        # Create dummy list of pairs of tomographic bins up to nbin (max of lenses and sources)
        all_calcs = [(i,j) for i in xrange(nbin) for j in xrange(nbin)]
        # Loop over tomographic bin pairs and add to final calculation list each requested correlation and unique healpix cell
        calcs=[]
        # Loop over needed shear-shear correlations
        for i,j in all_calcs:
            # Loop over tomographic bin pairs
            for pix_ in b1:
                # Loop over unique healpix cells
                if (i<=j) & (j<self.zbins) & (self.params['2pt_only'].lower() in [None,'shear-shear','all']):
                    calcs.append((i,j,pix_,0)) # Only add to list if calculating shear-shear and a valid tomographic pair (doesn't duplicate identical i<j and i>j)

        Npairs = dict()     
        if self.params['lens_group'] != 'None':
            for i,j in all_calcs:
                for pix_ in b1:
                    if (i<self.lens_zbins)&(j<self.zbins)&(self.params['2pt_only'].lower() in [None,'pos-shear','all']):
                        calcs.append((i,j,pix_,1))
                        
            for i,j in all_calcs:
                for pix_ in b1:
                    if (i<=j)&(j<self.lens_zbins)&(self.params['2pt_only'].lower() in [None,'pos-pos','all']):
                        calcs.append((i,j,pix_,2))
                if (i<=j)&(j<self.lens_zbins)&(self.params['2pt_only'].lower() in [None,'pos-pos','all']):
                     Npairs.update({'{0}_{1}'.format(i,j):self.compute_nobj(i,j)})
                        
        
        self.Npairs = Npairs
                       
        self.b1 = b1
        self.b = b
        self.all_calcs = all_calcs
        self.calcs=calcs
        
        # I should also load into memory ra,dec,weights and hpix..loading them for every jackknife region takes too long.
        bins_dict = dict()
        for i in xrange(nbin):
            # Loop over tomographic bin pairs
            if (i<self.zbins) & (self.params['2pt_only'].lower() in ['pos-shear','shear-shear','all']):
                ra,dec,g1,g2,w= self.build_catalogs_tot(self.source_calibrator,i)
                bins_dict.update({'shear_{0}'.format(i):[ra,dec,g1,g2,w]})
        if self.params['lens_group'] != 'None':
            for i in xrange(nbin):
                if (i<self.lens_zbins)&(self.params['2pt_only'].lower() in ['pos-pos','pos-shear','all']):
                    ra,dec,ran_ra,ran_dec,w,down = self.build_catalogs_tot(self.lens_calibrator,i)
                    bins_dict.update({'lens_{0}'.format(i):[ra,dec,ran_ra,ran_dec,w,down]})

        self.bins_dict = bins_dict         
                        
        
        if pool is None:
            self.rank=0
            self.size=1
        else:
            self.rank = pool.rank
            self.size = pool.size
        
        print ('done calcs')

        return calcs

    
    
    def get_weights(self,i):
        '''
        Get the weights column for the lens catalog and its random. Necessary for the computatuon of w(theta) with the jackknife module
        '''

        cal = self.lens_calibrator

        # Get tomographic bin masks for lenses and randoms, and weights
        R1,R2,mask,w_,rmask = self.get_zbins_R(i,cal)

        # Load ra,dec from gold catalog - source.read is necessary for the raw array to downmatch to lens catalog
        ra = self.lens_selector.source.read(self.Dict.lens_dict['ra'])[
            self.Dict.ind['u']][cal.selector.get_mask()[self.Dict.ind['u']]][mask]
        dec = self.lens_selector.source.read(self.Dict.lens_dict['dec'])[
            self.Dict.ind['u']][cal.selector.get_mask()[self.Dict.ind['u']]][mask]
        
        if not np.isscalar(w_):
            w   = w_[cal.selector.get_mask()[self.Dict.ind['u']]][mask]
        else:
            w = w_*np.ones(len(ra))
        
        if np.sum(rmask)>self.params['ran_factor']*np.sum(mask): 
            # Calculate if downsampling is possible
            # Set fixed random seed to make results reproducible
            np.random.seed(seed=self.params['random_seed'])
            # Downsample random catalog to be ran_factor times larger than lenses
            downsample = np.sort(np.random.choice(np.arange(np.sum(rmask)),self.params['ran_factor']*np.sum(mask),replace=False)) # Downsample 
        
        # Load random ra,dec and calculate healpix values
        ran_ra  = self.ran_selector.get_col(self.Dict.ran_dict['ra'])[self.Dict.ind['u']][rmask][downsample]
        ran_dec = self.ran_selector.get_col(self.Dict.ran_dict['dec'])[self.Dict.ind['u']][rmask][downsample]
        jck_fold = self.jack_dict_tot['{0}_lens'.format(i)]
        # _ , hpix_ran = jck_fold['centers_tree'].query(np.array(zip(ran_ra,ran_dec)))

        return jck_fold['hpix'],jck_fold['hpix_randoms'][downsample],w,np.ones(len(ran_ra))
        
        #return 0,0,w,np.ones(len(ran_ra))



    def compute_nobj(self,i,j):
        '''
        Compute tot number of objects in DD,DR,RD,RR. Necessary to compute w(theta) with the jackknife module
        '''
        
        mute = dict()
        pix_a,pixr_a, w_a,w_ra = self.get_weights(i)
        pix_b,pixr_b, w_b,w_rb = self.get_weights(j)
            
        NA,NB,NRA,NRB = np.sum(w_a),np.sum(w_b),np.sum(w_ra),np.sum(w_rb)
        n_jck  = self.jack_dict_tot['n_jck']

        def NN(ind,weight):
            lengths=np.zeros(n_jck)
            for i,n in enumerate(range(self.jack_dict_tot['n_jck'])):
                lengths[i]=np.sum(weight[ind==n])
            return lengths

        Na, Nb, Nra, Nrb = NN(pix_a,w_a), NN(pix_b,w_b), NN(pixr_a,w_ra), NN(pixr_b,w_rb)
        jck_N=np.zeros((n_jck,4))
        jck_N[:,0]= Na
        jck_N[:,1]= Nb
        jck_N[:,2]= Nra
        jck_N[:,3]= Nrb
        N_A, N_B, N_RA, N_RB = NA-np.array(Na), NB-np.array(Nb), NRA-np.array(Nra), NRB-np.array(Nrb)
        mute.update({'N':  [N_A*N_B, N_A*N_RB, N_RA*N_B, N_RA*N_RB]})
        mute.update({'jck_N':jck_N})
        return mute
    
    def run(self):
        """
        This is where all the calculations are done. A mpi pool is set up and the jobs accumulated in setup_jobs() are distributed across nodes.
        
        """

        # run full ************************************     
        if (self.params['region_mode'] == 'full') or (self.params['region_mode'] == 'both'):
            if self.comm:
                # Parallel execution
                from .mpi_pool import MPIPool
                # Setup mpi pool
                pool = MPIPool(self.comm)
                # Get job list
                calcs = self.setup_jobs(pool)
                self.comm.Barrier()
            
                # Master distributes calculations across nodes.
               
                pool.map(task_full, calcs)

                print ('out of main loop',pool.rank)
                sys.stdout.flush()
                pool.close()
            else:
                # Serial execution
                calcs = self.setup_jobs(None)
                map(task_full, calcs)
                
        if (self.params['region_mode'] == 'pixellized') or (self.params['region_mode'] == 'both'):
            if self.comm:
                # Parallel execution
                from .mpi_pool import MPIPool
                # Setup mpi pool
                pool = MPIPool(self.comm)
                # Get job list
                calcs = self.setup_jobs(pool)
                self.comm.Barrier()
            
                # Master distributes calculations across nodes.
               
                pool.map(task, calcs)

                print ('out of main loop',pool.rank)
                sys.stdout.flush()
                pool.close()
            else:
                # Serial execution
                calcs = self.setup_jobs(None)
                map(task, calcs)

    def get_zbins_R(self,i,cal):
        """
        Get the lens or source binning, calibration, and weights for a given tomographic bin.
        """ 
        
        # Open file from nofz stage that contains the catalog tomographic binning indices and read.
        f = h5py.File( self.input_path("nz_source"), mode='r')
        if (type(cal)==destest.NoCalib) and (cal.params['cal_type'] is None):
            binning = [f['nofz/lens_zbin'][:]]
        else: # Source catalog
            binning = []
            if self.params['has_sheared']:
                zbins = ['zbin', 'zbin_1p', 'zbin_1m', 'zbin_2p', 'zbin_2m']
            else:
                zbins = ['zbin']

            for zbin_ in zbins: # Length 5 for unsheared and sheared metacal selections
                binning.append(f['nofz'][zbin_][:])

        # Create tomographic binning mask
        mask = []
        for s in binning:
            mask.append( s == i )

        if (type(cal)==destest.NoCalib) and (cal.params['cal_type'] is None): # Lens catalog

            # Get random binning as well
            f = h5py.File( self.input_path("nz_source"), mode='r')
            rmask = f['nofz']['ran_zbin'][:] == i
            # Get weights
            w = cal.calibrate(self.Dict.shape_dict['e1'], weight_only=True)
            # Return lens binning mask, weights, and random binning mask
            return np.ones_like(w), np.ones_like(w), mask[self.Dict.ind['u']], w, rmask

        elif (type(cal)==destest.NoCalib): # buzzard catalog

            w = cal.calibrate(self.Dict.shape_dict['e1'], weight_only=True)
            return np.ones_like(w), np.ones_like(w), mask[self.Dict.ind['u']], w

        else: # Source catalog

            # Get responses
            R1,c,w = cal.calibrate(self.Dict.shape_dict['e1'], mask=mask)
            R2,c,w = cal.calibrate(self.Dict.shape_dict['e2'], mask=mask)
            # Return responses, source binning mask and weights            
            return R1, R2, mask[self.Dict.ind['u']], w


    def build_catalogs_tot(self,cal,i):
        """
        Buid catalog subsets in the form of treecorr.Catalog objects for the required tomograhpic and healpixel subsets for this calculation iteration.
        """
        start = timeit.default_timer()
        if (type(cal)==destest.NoCalib) and (cal.params['cal_type'] is None): # lens catalog


            # Get tomographic bin masks for lenses and randoms, and weights
            R1,R2,mask,w_,rmask = self.get_zbins_R(i,cal)

            # Load ra,dec from gold catalog - source.read is necessary for the raw array to downmatch to lens catalog
            ra = self.lens_selector.source.read(self.Dict.lens_dict['ra'])[self.Dict.ind['u']][cal.selector.get_mask()[self.Dict.ind['u']]][mask]
            dec = self.lens_selector.source.read(self.Dict.lens_dict['dec'])[self.Dict.ind['u']][cal.selector.get_mask()[self.Dict.ind['u']]][mask]
            
            if not np.isscalar(w_):
                w   = w_[cal.selector.get_mask()[self.Dict.ind['u']]][mask]
            else:
                w = w_*np.ones(len(ra))
            
            if np.sum(rmask)>self.params['ran_factor']*np.sum(mask): # Calculate if downsampling is possible
                # Set fixed random seed to make results reproducible
                np.random.seed(seed=self.params['random_seed'])
                # Downsample random catalog to be ran_factor times larger than lenses
                downsample = np.sort(np.random.choice(np.arange(np.sum(rmask)),self.params['ran_factor']*np.sum(mask),replace=False)) # Downsample 

            # Load random ra,dec and calculate healpix values
            ran_ra  = self.ran_selector.get_col(self.Dict.ran_dict['ra'])[self.Dict.ind['u']][rmask][downsample]
            ran_dec = self.ran_selector.get_col(self.Dict.ran_dict['dec'])[self.Dict.ind['u']][rmask][downsample]
            end =  timeit.default_timer()
            
            print ("full load ", end-start)
            return ra,dec,ran_ra,ran_dec,w,downsample

        else: # source catalog

            # Get tomographic bin masks for sources, and responses/weights
            R1,R2,mask,w_ = self.get_zbins_R(i,cal)

            # Load ra,dec from gold catalog
            ra=self.gold_selector.get_col(self.Dict.gold_dict['ra'])[self.Dict.ind['u']][mask]
            dec=self.gold_selector.get_col(self.Dict.gold_dict['dec'])[self.Dict.ind['u']][mask]

            # Get e1,e2, subtract mean shear, and correct with mean response
            g1=cal.selector.get_col(self.Dict.shape_dict['e1'])[self.Dict.ind['u']][mask]
            print('----------',g1,self.mean_e1[i],R1)
            g1 = (g1-self.mean_e1[i])/R1
            g2=cal.selector.get_col(self.Dict.shape_dict['e2'])[self.Dict.ind['u']][mask]
            g2 = (g2-self.mean_e2[i])/R2
            if self.params['flip_e2']==True:
                print('flipping e2')
                g2*=-1

            w = w_*np.ones(len(ra))
            
            end =  timeit.default_timer()
            
            print ("full load " ,end-start)
            return ra,dec,g1,g2,w

    def calc_correlation(self,i,j,k,pix1=None,pix2=None,full=False):
        start =  timeit.default_timer()
        """
        Treecorr wrapper for shear-shear,shear-pos,pos-pos calculations.
        """
        
        #print ("slope",self.params['slop'])
        num_threads = self.params['cores_per_task']

        if k ==0:
            type_corr = 'shear_shear'

            # shear cat i
            jck_fold = self.jack_dict_tot['{0}_shears'.format(i)]
            if not full:
                mask_jck = np.in1d(jck_fold['hpix'],pix1)
                ra,dec,g1,g2,w = self.bins_dict['shear_{0}'.format(i)][0][mask_jck],self.bins_dict['shear_{0}'.format(i)][1][mask_jck],self.bins_dict['shear_{0}'.format(i)][2][mask_jck],self.bins_dict['shear_{0}'.format(i)][3][mask_jck],self.bins_dict['shear_{0}'.format(i)][4][mask_jck]
            else:
                ra,dec,g1,g2,w = self.bins_dict['shear_{0}'.format(i)]
                
           
            try:
                icat = treecorr.Catalog( g1 = g1, g2   = g2, 
                                 ra = ra, dec  = dec, 
                                 w  = w, wpos = np.ones(len(ra)), 
                                 ra_units='deg', dec_units='deg')
            except:
                icat = None
                
            # shear cat j 
            jck_fold = self.jack_dict_tot['{0}_shears'.format(j)]
            if not full:
                mask_jck = np.in1d(jck_fold['hpix'],pix2)
                ra,dec,g1,g2,w = self.bins_dict['shear_{0}'.format(j)][0][mask_jck],self.bins_dict['shear_{0}'.format(j)][1][mask_jck],self.bins_dict['shear_{0}'.format(j)][2][mask_jck],self.bins_dict['shear_{0}'.format(j)][3][mask_jck],self.bins_dict['shear_{0}'.format(j)][4][mask_jck]
            else:
                ra,dec,g1,g2,w = self.bins_dict['shear_{0}'.format(j)]
               
            try:
                jcat = treecorr.Catalog( g1 = g1, g2   = g2, 
                                 ra = ra, dec  = dec, 
                                 w  = w, wpos = np.ones(len(ra)), 
                                 ra_units='deg', dec_units='deg')
            except:
                jcat = None
                
            if (icat == None) or (jcat == None):
                pairs = [np.zeros(self.params['tbins']), np.zeros(self.params['tbins']), np.zeros(self.params['tbins']),np.zeros(self.params['tbins']),np.zeros(self.params['tbins']), np.zeros(self.params['tbins']), np.zeros(self.params['tbins']),np.zeros(self.params['tbins'])]
            else:
                
        
                gg = treecorr.GGCorrelation(nbins=self.params['tbins'], min_sep=self.params['tbounds'][self.Dict.ind['u']], max_sep=self.params['tbounds'][1], sep_units='arcmin', bin_slop=self.params['slop'], num_threads=num_threads)
                gg.process(icat, jcat)
                ggp = gg.xip
                ggm = gg.xim
                normalization = gg.weight
                 
                pairs = [ggp * normalization, ggm * normalization, normalization, gg.npairs,gg.logr*gg.weight,gg.rnom*gg.weight,gg.meanr*gg.weight,gg.meanlogr*gg.weight]
                
            end =  timeit.default_timer()
            #print "shear comp ", end-start
            return pairs

        
        
        elif k ==1:
            type_corr = 'shear_pos'
            
            
            
                
            # pos cat i (+random!)
            jck_fold = self.jack_dict_tot['{0}_lens'.format(i)]
            if not full:
                mask_jck = np.in1d(jck_fold['hpix'],pix1)
                downsample = [self.bins_dict['lens_{0}'.format(i)][5]]
                
                mask_jck_rndm = np.in1d(jck_fold['hpix_randoms'][downsample],pix1)
                
                ra,dec,ran_ra,ran_dec,w = self.bins_dict['lens_{0}'.format(i)][0][mask_jck],self.bins_dict['lens_{0}'.format(i)][1][mask_jck],self.bins_dict['lens_{0}'.format(i)][2][mask_jck_rndm],self.bins_dict['lens_{0}'.format(i)][3][mask_jck_rndm],self.bins_dict['lens_{0}'.format(i)][4]
                
               
            else:
                ra,dec,ran_ra,ran_dec,w,_ = self.bins_dict['lens_{0}'.format(i)]
                
            try:
                icat = treecorr.Catalog(ra = ra, dec  = dec, 
                                 w  = w, wpos = np.ones(len(ra)), 
                                 ra_units='deg', dec_units='deg')
                
            except:
                icat = None
            try:
          
                ircat = treecorr.Catalog(ra = ran_ra, dec  = ran_dec, 
                                 w  = np.ones(len(ran_ra)), wpos = np.ones(len(ran_ra)), 
                                 ra_units='deg', dec_units='deg')
            except:
                ircat = None
            
            # shear cat j 
            jck_fold = self.jack_dict_tot['{0}_shears'.format(j)]
            if not full:
                mask_jck = np.in1d(jck_fold['hpix'],pix2)
                ra,dec,g1,g2,w = self.bins_dict['shear_{0}'.format(j)][0][mask_jck],self.bins_dict['shear_{0}'.format(j)][1][mask_jck],self.bins_dict['shear_{0}'.format(j)][2][mask_jck],self.bins_dict['shear_{0}'.format(j)][3][mask_jck],self.bins_dict['shear_{0}'.format(j)][4][mask_jck]
            else:
                ra,dec,g1,g2,w = self.bins_dict['shear_{0}'.format(j)]
               
     
            try:
                jcat = treecorr.Catalog( g1 = g1, g2   = g2, 
                                 ra = ra, dec  = dec, 
                                 w  = w, wpos = np.ones(len(ra)), 
                                 ra_units='deg', dec_units='deg')
                
            except:
                jcat = None

            
            pairs = [np.zeros(self.params['tbins']), np.zeros(self.params['tbins']), np.zeros(self.params['tbins']), np.zeros(self.params['tbins']),np.zeros(self.params['tbins']), np.zeros(self.params['tbins']),np.zeros(self.params['tbins']), np.zeros(self.params['tbins']),np.zeros(self.params['tbins']), np.zeros(self.params['tbins']), np.zeros(self.params['tbins']),np.zeros(self.params['tbins']),np.zeros(self.params['tbins']),np.zeros(self.params['tbins'])]
            
            ng = treecorr.NGCorrelation(nbins=self.params['tbins'], min_sep=self.params['tbounds'][self.Dict.ind['u']], max_sep=self.params['tbounds'][1], sep_units='arcmin', bin_slop=self.params['slop'],num_threads=num_threads)
            rg = treecorr.NGCorrelation(nbins=self.params['tbins'], min_sep=self.params['tbounds'][self.Dict.ind['u']], max_sep=self.params['tbounds'][1], sep_units='arcmin', bin_slop=self.params['slop'], num_threads=num_threads)

            if (icat != None) and (jcat != None):
                ng.process(icat, jcat)
                xi = ng.xi
                xi_im = ng.xi_im
                normalization_n = ng.weight


                pairs[0] = xi * normalization_n
                pairs[2] = normalization_n
                pairs[4] = xi_im * normalization_n
                pairs[6] = ng.npairs

                pairs[10] = ng.logr*ng.weight
                pairs[11] = ng.rnom*ng.weight
                pairs[12] = ng.meanr*ng.weight
                pairs[13] = ng.meanlogr*ng.weight
            if (ircat != None) and (jcat != None):
                rg.process(ircat, jcat)
                if (icat != None) and (jcat != None):
                    gammat,gammat_im,gammaterr=ng.calculateXi(rg)
                    pairs[8] = gammat * normalization_n
                    pairs[9] = gammat_im*normalization_n
                xi_r = rg.xi
                xi_im_r = rg.xi_im
                normalization_r = rg.weight
                pairs[1] =  xi_r * normalization_r
                pairs[3] = normalization_r
                pairs[5] = xi_im_r * normalization_r
                pairs[7] = rg.npairs
                
            end =  timeit.default_timer()
            #print "shear comp ", end-start
            return pairs
        
        elif k ==2:
            type_corr = 'pos_pos'
            
            # pos cat i (+random!)
            jck_fold = self.jack_dict_tot['{0}_lens'.format(i)]
            if not full:
                mask_jck = np.in1d(jck_fold['hpix'],pix1)
                downsample = self.bins_dict['lens_{0}'.format(i)][5]
                mask_jck_rndm = np.in1d(jck_fold['hpix_randoms'][downsample],pix1)
                
                ra,dec,ran_ra,ran_dec,w = self.bins_dict['lens_{0}'.format(i)][0][mask_jck],self.bins_dict['lens_{0}'.format(i)][1][mask_jck],self.bins_dict['lens_{0}'.format(i)][2][mask_jck_rndm],self.bins_dict['lens_{0}'.format(i)][3][mask_jck_rndm],self.bins_dict['lens_{0}'.format(i)][4][mask_jck]
            else:
                ra,dec,ran_ra,ran_dec,w,_ = self.bins_dict['lens_{0}'.format(i)]
            try:
                icat = treecorr.Catalog(ra = ra, dec  = dec, 
                                 w  = w, wpos = np.ones(len(ra)), 
                                 ra_units='deg', dec_units='deg')
            except:
                icat = None
            try:
                ircat = treecorr.Catalog(ra = ran_ra, dec  = ran_dec, 
                                 w  = np.ones(len(ran_ra)), wpos = np.ones(len(ran_ra)), 
                                 ra_units='deg', dec_units='deg')
            except:
                ircat = None
            
            # pos cat j (+random!)
            jck_fold = self.jack_dict_tot['{0}_lens'.format(j)]
            if not full:
                mask_jck = np.in1d(jck_fold['hpix'],pix2)
                downsample = [self.bins_dict['lens_{0}'.format(j)][5]]
                mask_jck_rndm = np.in1d(jck_fold['hpix_randoms'][downsample],pix2)
                ra,dec,ran_ra,ran_dec,w = self.bins_dict['lens_{0}'.format(j)][0][mask_jck],self.bins_dict['lens_{0}'.format(j)][1][mask_jck],self.bins_dict['lens_{0}'.format(j)][2][mask_jck_rndm],self.bins_dict['lens_{0}'.format(j)][3][mask_jck_rndm],self.bins_dict['lens_{0}'.format(j)][4][mask_jck]
            else:
                ra,dec,ran_ra,ran_dec,w,_ = self.bins_dict['lens_{0}'.format(j)]
            try:
                jcat = treecorr.Catalog(ra = ra, dec  = dec, 
                                 w  = w, wpos = np.ones(len(ra)), 
                                 ra_units='deg', dec_units='deg')
            except:
                jcat = None
            try:
                jrcat = treecorr.Catalog(ra = ran_ra, dec  = ran_dec, 
                                 w  = np.ones(len(ran_ra)), wpos = np.ones(len(ran_ra)), 
                                 ra_units='deg', dec_units='deg')
            except:
                jrcat = None
                
            
            pairs = [np.zeros(self.params['tbins']), np.zeros(self.params['tbins']), np.zeros(self.params['tbins']), np.zeros(self.params['tbins']),np.zeros(self.params['tbins']), np.zeros(self.params['tbins']), np.zeros(self.params['tbins']), np.zeros(self.params['tbins']),np.zeros(self.params['tbins']),np.zeros(self.params['tbins']),np.zeros(self.params['tbins']),np.zeros(self.params['tbins'])]
            
            dd = treecorr.NNCorrelation(nbins=self.params['tbins'], min_sep=self.params['tbounds'][self.Dict.ind['u']], max_sep=self.params['tbounds'][1], sep_units='arcmin', bin_slop=self.params['slop'], num_threads=num_threads)
            dr = treecorr.NNCorrelation(nbins=self.params['tbins'], min_sep=self.params['tbounds'][self.Dict.ind['u']], max_sep=self.params['tbounds'][1], sep_units='arcmin', bin_slop=self.params['slop'],num_threads=num_threads)
            rd = treecorr.NNCorrelation(nbins=self.params['tbins'], min_sep=self.params['tbounds'][self.Dict.ind['u']], max_sep=self.params['tbounds'][1], sep_units='arcmin', bin_slop=self.params['slop'], num_threads=num_threads)
            rr = treecorr.NNCorrelation(nbins=self.params['tbins'], min_sep=self.params['tbounds'][self.Dict.ind['u']], max_sep=self.params['tbounds'][1], sep_units='arcmin', bin_slop=self.params['slop'], num_threads=num_threads)
            
            if (icat!=None )and(jcat!=None):
                dd.process(icat, jcat)
                pairs[0] = dd.weight
                pairs[4] = dd.npairs
                pairs[8] = dd.logr*dd.weight
                pairs[9] = dd.rnom*dd.weight
                pairs[10] = dd.meanr*dd.weight
                pairs[11] = dd.meanlogr*dd.weight
            if (icat!=None )and(jrcat!=None):
                dr.process(icat, jrcat)
                pairs[1] = dr.weight
                pairs[5] = dr.npairs
            if (ircat!=None )and(jcat!=None):  
                rd.process(ircat, jcat)
                pairs[2] = rd.weight
                pairs[6] = rd.npairs
            if (ircat!=None )and(jrcat!=None):
                rr.process(ircat, jrcat)
                pairs[3] = rr.weight
                pairs[7] = rr.npairs
            

            
            end =  timeit.default_timer()
            #print "shear comp " ,end-start
            return pairs   
            
        elif k ==3:
            type_corr = 'kappa_kappa'
            
        return 

  
    def write(self):
        import numpy as np
        import os
        
        def collect(pairs,n_jck,n_bins,type_corr,i,j):
            shape = (n_jck, n_bins)
            DD_a, DR_a, RD_a, RR_a,mm1_a,mm2_a,mm3_a,mm4_a,mm5_a,mm6_a,mm7_a,mm8_a,mm9_a,mm10_a = np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape),np.zeros(shape), np.zeros(shape),np.zeros(shape), np.zeros(shape),np.zeros(shape),np.zeros(shape), np.zeros(shape),np.zeros(shape),np.zeros(shape),np.zeros(shape)
            DD, DR, RD, RR,mm1,mm2,mm3,mm4,mm5,mm6,mm7,mm8,mm9,mm10 = np.zeros(n_bins),  np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins),np.zeros(n_bins), np.zeros(n_bins),np.zeros(n_bins), np.zeros(n_bins),np.zeros(n_bins), np.zeros(n_bins),np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)


            fact = 1.
            FACT_0 = 0.5

            for n in range(n_bins):
                for jk1 in range(len(pairs[:, 0, 0, n])):
                    DD[n] += (pairs[jk1, 0, 0, n]) + FACT_0 * (pairs[jk1, 1, 0, n])
                    DR[n] += ((pairs[jk1, 0, 1, n]) + FACT_0 * (pairs[jk1, 1, 1, n]))
                    #print DR, DD, RD
                    RD[n] += ((pairs[jk1, 0, 2, n]) + FACT_0 * (pairs[jk1, 1, 2, n]))
                    try:
                        RR[n] += ((pairs[jk1, 0, 3, n]) + FACT_0 * (pairs[jk1, 1, 3, n]))
                    except:
                        pass
                    try:
                        mm1[n]+= ((pairs[jk1, 0, 4, n]) + FACT_0 * (pairs[jk1, 1, 4, n]))
                    except:
                        pass
                    try:
                        mm2[n]+= ((pairs[jk1, 0, 5, n]) + FACT_0 * (pairs[jk1, 1, 5, n]))
                    except:
                        pass
                    try:
                        mm3[n]+= ((pairs[jk1, 0, 6, n]) + FACT_0 * (pairs[jk1, 1, 6, n]))
                    except:
                        pass
                    try:
                        mm4[n]+= ((pairs[jk1, 0, 7, n]) + FACT_0 * (pairs[jk1, 1, 7, n]))
                    except:
                        pass
                    try:
                        mm5[n]+= ((pairs[jk1, 0, 8, n]) + FACT_0 * (pairs[jk1, 1, 8, n]))
                    except:
                        pass
                    try:
                        mm6[n]+= ((pairs[jk1, 0, 9, n]) + FACT_0 * (pairs[jk1, 1, 9, n]))
                    except:
                        pass
                    try:
                        mm7[n]+= ((pairs[jk1, 0, 10, n]) + FACT_0 * (pairs[jk1, 1, 10, n]))
                    except:
                        pass
                    try:
                        mm8[n]+= ((pairs[jk1, 0, 11, n]) + FACT_0 * (pairs[jk1, 1, 11, n]))
                    except:
                        pass    
                    try:
                        mm9[n]+= ((pairs[jk1, 0, 12, n]) + FACT_0 * (pairs[jk1, 1, 12, n]))
                    except:
                        pass    
                    try:
                        mm10[n]+= ((pairs[jk1, 0, 13, n]) + FACT_0 * (pairs[jk1, 1, 13, n]))
                    except:
                        pass          
            for n in range(n_bins):
                for jk1 in range(len(pairs[:, 0, 0, n])):
                    DD_a[jk1, n] = DD[n] - (pairs[jk1, 0, 0, n]) - fact * FACT_0 * (pairs[jk1, 1, 0, n])
                    DR_a[jk1, n] = DR[n] - (pairs[jk1, 0, 1, n]) - fact * FACT_0 * (pairs[jk1, 1, 1, n])
                    RD_a[jk1, n] = RD[n] - (pairs[jk1, 0, 2, n]) - fact * FACT_0 * (pairs[jk1, 1, 2, n])
                    try:
                        RR_a[jk1, n] = RR[n] - (pairs[jk1, 0, 3, n]) - fact * FACT_0 * (pairs[jk1, 1, 3, n])
                    except:
                        pass
                    try:
                        mm1_a[jk1, n] = mm1[n] - (pairs[jk1, 0, 4, n]) - fact * FACT_0 * (pairs[jk1, 1, 4, n])
                    except:
                        pass
                    try:
                        mm2_a[jk1, n] = mm2[n] - (pairs[jk1, 0, 5, n]) - fact * FACT_0 * (pairs[jk1, 1, 5, n])
                    except:
                        pass
                    try:
                        mm3_a[jk1, n] = mm3[n] - (pairs[jk1, 0, 6, n]) - fact * FACT_0 * (pairs[jk1, 1, 6, n])
                    except:
                        pass
                    try:
                        mm4_a[jk1, n] = mm4[n] - (pairs[jk1, 0, 7, n]) - fact * FACT_0 * (pairs[jk1, 1, 7, n])
                    except:
                        pass             
                    try:
                        mm5_a[jk1, n] = mm5[n] - (pairs[jk1, 0, 8, n]) - fact * FACT_0 * (pairs[jk1, 1, 8, n])
                    except:
                        pass
                    try:
                        mm6_a[jk1, n] = mm6[n] - (pairs[jk1, 0, 9, n]) - fact * FACT_0 * (pairs[jk1, 1, 9, n])
                    except:
                        pass
                    try:
                        mm7_a[jk1, n] = mm7[n] - (pairs[jk1, 0, 10, n]) - fact * FACT_0 * (pairs[jk1, 1, 10, n])
                    except:
                        pass
                    try:
                        mm8_a[jk1, n] = mm8[n] - (pairs[jk1, 0, 11, n]) - fact * FACT_0 * (pairs[jk1, 1, 11, n])
                    except:
                        pass   
                    try:
                        mm9_a[jk1, n] = mm9[n] - (pairs[jk1, 0, 12, n]) - fact * FACT_0 * (pairs[jk1, 1, 12, n])
                    except:
                        pass  
                    try:
                        mm10_a[jk1, n] = mm10[n] - (pairs[jk1, 0, 13, n]) - fact * FACT_0 * (pairs[jk1, 1, 13, n])
                    except:
                        pass 
                    #print RD_a
            if (type_corr== 'shear_shear' ):
                xip = np.zeros(len(RD))
                xim = np.zeros(len(RD))
                logr =  np.zeros(len(RD))
                rnom =  np.zeros(len(RD))
                meanr = np.zeros(len(RD))
                meanlogr = np.zeros(len(RD))
                masku = RD !=0.
            
                xip[masku] = DD[masku] / RD[masku]
                xim[masku] = DR[masku] / RD[masku]

                masku = RD_a !=0.
                xip_j = np.zeros((RD_a.shape[0],RD_a.shape[1]))
                xim_j = np.zeros((RD_a.shape[0],RD_a.shape[1]))
                xip_j[masku] = DD_a[masku] / RD_a[masku]
                xim_j[masku] = DR_a[masku] / RD_a[masku]
                npairs = RD
                npairs_j = RD_a
                weight = RR
                weight_j = RR_a

                masku = RD !=0.
                logr[masku] = mm1[masku]/ RD[masku]
                rnom[masku] = mm2[masku]/ RD[masku]
                meanr[masku] = mm3[masku]/ RD[masku]
                meanlogr[masku] = mm4[masku]/RD[masku]
                

                return xip, xim, xip_j, xim_j,npairs,weight,logr,rnom,meanr,meanlogr
            
            
            if type_corr== 'pos_pos' :
                
                
                ndd=(np.sum(global_measure_2_point.Npairs['{0}_{1}'.format(i,j)]['jck_N'][:,0]))*(np.sum(global_measure_2_point.Npairs['{0}_{1}'.format(i,j)]['jck_N'][:,1]))
                ndr=(np.sum(global_measure_2_point.Npairs['{0}_{1}'.format(i,j)]['jck_N'][:,0]))*(np.sum(global_measure_2_point.Npairs['{0}_{1}'.format(i,j)]['jck_N'][:,3]))
                nrd=(np.sum(global_measure_2_point.Npairs['{0}_{1}'.format(i,j)]['jck_N'][:,2]))*(np.sum(global_measure_2_point.Npairs['{0}_{1}'.format(i,j)]['jck_N'][:,1]))
                nrr=(np.sum(global_measure_2_point.Npairs['{0}_{1}'.format(i,j)]['jck_N'][:,2]))*(np.sum(global_measure_2_point.Npairs['{0}_{1}'.format(i,j)]['jck_N'][:,3]))
                    
                norm=[1.,ndd/ndr,ndd/nrd,ndd/nrr]

                xi = np.zeros(len(RD))
                logr =  np.zeros(len(RD))
                rnom =  np.zeros(len(RD))
                meanr = np.zeros(len(RD))
                meanlogr = np.zeros(len(RD))
                xi_j = np.zeros((RD_a.shape[0],RD_a.shape[1]))
                masku = RR !=0.
            
                xi[masku] = (DD[masku]-DR[masku]*norm[1]-RD[masku]*norm[2]+RR[masku]*norm[3]) / (RR[masku]*norm[3])

                
                
                
                for jk in range(self.jack_dict_tot['n_jck']):
                    ndd=(np.sum(self.Npairs['{0}_{1}'.format(i,j)]['jck_N'][:,0])-self.Npairs['{0}_{1}'.format(i,j)]['jck_N'][jk,0])*(np.sum(self.Npairs['{0}_{1}'.format(i,j)]['jck_N'][:,1])-self.Npairs['{0}_{1}'.format(i,j)]['jck_N'][jk,1])
                    ndr=(np.sum(self.Npairs['{0}_{1}'.format(i,j)]['jck_N'][:,0])-self.Npairs['{0}_{1}'.format(i,j)]['jck_N'][jk,0])*(np.sum(self.Npairs['{0}_{1}'.format(i,j)]['jck_N'][:,3])-self.Npairs['{0}_{1}'.format(i,j)]['jck_N'][jk,3])
                    nrd=(np.sum(self.Npairs['{0}_{1}'.format(i,j)]['jck_N'][:,2])-self.Npairs['{0}_{1}'.format(i,j)]['jck_N'][jk,2])*(np.sum(self.Npairs['{0}_{1}'.format(i,j)]['jck_N'][:,1])-self.Npairs['{0}_{1}'.format(i,j)]['jck_N'][jk,1])
                    nrr=(np.sum(self.Npairs['{0}_{1}'.format(i,j)]['jck_N'][:,2])-self.Npairs['{0}_{1}'.format(i,j)]['jck_N'][jk,2])*(np.sum(self.Npairs['{0}_{1}'.format(i,j)]['jck_N'][:,3])-self.Npairs['{0}_{1}'.format(i,j)]['jck_N'][jk,3])
                    norm=np.array([1.,ndd/ndr,ndd/nrd,ndd/nrr])
                    
                    masku = RR_a[jk,:] !=0.
                    xi_j[jk,masku] = (DD_a[jk,masku]-DR_a[jk,masku]*norm[1] -RD_a[jk,masku]*norm[2] +RR_a[jk,masku]*norm[3] ) /( RR_a[jk,masku]*norm[3])

                npairs = mm1
                weight = DD
                
                masku = DD !=0.
                logr[masku] = mm5[masku]/ DD[masku]
                rnom[masku] = mm6[masku]/ DD[masku]
                meanr[masku] = mm7[masku]/ DD[masku]
                meanlogr[masku] = mm8[masku]/ DD[masku]
                
                return xi,xi_j,npairs,weight,logr,rnom,meanr,meanlogr

            if type_corr== 'shear_pos' :
                xi = np.zeros(len(RD))
                xir = np.zeros(len(RD))
                xi_im = np.zeros(len(RD))
                xi_imr = np.zeros(len(RD))
                
                
                masku = RD !=0.
                xi[masku] = DD[masku]/ RD[masku]
                xi_im[masku] = mm1[masku]/ RD[masku]

               
                
                gammat_mr=np.zeros(len(RD))
                gammat_mr_im =np.zeros(len(RD))
                gammat_mr[masku] = mm5[masku]/ RD[masku]
                gammat_mr_im[masku] = mm6[masku]/ RD[masku]
                
                logr =  np.zeros(len(RD))
                rnom =  np.zeros(len(RD))
                meanr = np.zeros(len(RD))
                meanlogr = np.zeros(len(RD))
                logr[masku] = mm7[masku]/ RD[masku]
                rnom[masku] = mm8[masku]/ RD[masku]
                meanr[masku] = mm9[masku]/ RD[masku]
                meanlogr[masku] = mm10[masku]/ RD[masku]
                
                
                masku = RR !=0.
                xir[masku] = DR[masku]/ RR[masku]
                xi_imr[masku] = mm2[masku]/ RR[masku]
                
                xi_j = np.zeros((RD_a.shape[0],RD_a.shape[1]))
                xir_j = np.zeros((RD_a.shape[0],RD_a.shape[1]))
                xi_im_j = np.zeros((RD_a.shape[0],RD_a.shape[1]))
                xi_imr_j = np.zeros((RD_a.shape[0],RD_a.shape[1]))
                gammat_mr_j = np.zeros((RD_a.shape[0],RD_a.shape[1]))
                gammat_mr_im_j = np.zeros((RD_a.shape[0],RD_a.shape[1]))
                
                masku = RD_a !=0.
                xi_j[masku] = (DD_a[masku]) / RD_a[masku]
                xi_im_j[masku] = (mm1_a[masku]) / RD_a[masku]
                masku = RR_a !=0.
                xir_j[masku] = (DR_a[masku]) / RR_a[masku]
                xi_imr_j[masku] = (mm2_a[masku]) / RR_a[masku]
                
                masku = RD_a !=0.
                gammat_mr_j[masku] =(mm5_a[masku]) / RD_a[masku]
                gammat_mr_im_j[masku] =(mm6_a[masku]) / RD_a[masku]
                
                npairs = mm3
                npairs_j = mm3_a
                
                npairs_im = mm4
                npairs_im_j = mm4_a
                
                weight = RD
                weight_j = RD_a
                weight_im = RR
                weight_im_j = RR_a  
                
      

                
  
                
                return xi,xi_j,xir,xir_j,xi_im,xi_im_j,xi_imr,xi_imr_j,npairs,npairs_im,weight,weight_im,gammat_mr,gammat_mr_im,gammat_mr_j,gammat_mr_im_j,logr,rnom,meanr,meanlogr
        
        
        
        """
        Write data to files - Collect 2pt
        """
        # Get max number of tomographic bins between lenses and sources
        if self.params['lens_group'] != 'None':
            nbin=max(self.lens_zbins,self.zbins)
        else:
            nbin=self.zbins

        output_shear_shear = dict()
        output_shear_pos = dict()
        output_pos_pos = dict()
        
        output_shear_shear_full = dict()
        output_shear_pos_full = dict()
        output_pos_pos_full = dict()
        
        for i,j in self.all_calcs:
            # Loop over unique healpix cells
            if (i<=j) & (j<self.zbins) & (self.params['2pt_only'].lower() in [None,'shear-shear','all']):
                # Loop over tomographic bin pairs
                
                if (self.params['region_mode'] == 'pixellized') or (self.params['region_mode'] == 'both'):
                    gm = 3+1+4
                    shape = (gm, self.params['tbins'])
                    pairs_ring = [[np.zeros(shape) for ii in range(2)] for jk in range(self.jack_dict_tot['n_jck'])]
                    path = self.params['run_directory']+'/2pt/{0}_{1}_{2}/'.format(i,j,0)
                    for jci,jc in enumerate(np.unique(self.b)):
                        print(i,j,jc)
                        lla = '{0}'.format(jc)
                        dict_m = load_obj(path+lla)
                        pairsCC1 = dict_m['c1'][:gm]
                        pairsCC2 = dict_m['c2'][:gm]
                        pairs_auto = dict_m['a'][:gm]
                        for prs in range(gm):
                            pairsCC1[prs] += pairsCC2[prs]
                    
                        pairs_ring[jci][1] = pairsCC1
                        pairs_ring[jci][0] = pairs_auto  
                    
                    pairs_ring = np.array(pairs_ring)
                    self.pairs_ring = pairs_ring
                    xip, xim, xip_j, xim_j,npairs,weight,logr,rnom,meanr,meanlogr = collect(pairs_ring,self.jack_dict_tot['n_jck'],self.params['tbins'],'shear_shear',i,j)
                    cov_xip = covariance_jck( xip_j.T,self.jack_dict_tot['n_jck'],'jackknife')
                    cov_xim = covariance_jck( xim_j.T,self.jack_dict_tot['n_jck'],'jackknife')
                    files = dict()
                    files.update({'logr':logr})
                    files.update({'rnom':rnom})
                    files.update({'meanr':meanr})
                    files.update({'meanlogr':meanlogr})   
                    files.update({'xip':xip})
                    files.update({'xip_jack':xip_j})
                    files.update({'cov_xip_jack':cov_xip})
                    files.update({'xim':xim})
                    files.update({'xim_jack':xim_j})
                    files.update({'cov_xim_jack':cov_xim})
                    files.update({'npairs':npairs})
                    files.update({'weight':weight})
                    
                    save_obj(path+'/results',files)
                    output_shear_shear.update({'{0}_{1}'.format(i,j):files})
                
                if (self.params['region_mode'] == 'full') or (self.params['region_mode'] == 'both'):
                    path = self.params['run_directory']+'/2pt/{0}_{1}_{2}/_full'.format(i,j,0)
                    if os.path.exists(path+'.pkl'):
                        mute = load_obj(path)
                        xip_full = (mute[0]/mute[2])
                        xim_full = (mute[1]/mute[2])
                        muted = dict()
                        muted.update({'logr':mute[4]/mute[2]})
                        muted.update({'rnom':mute[5]/mute[2]})
                        muted.update({'meanr':mute[6]/mute[2]})
                        muted.update({'meanlogr':mute[7]/mute[2]})  
                        muted.update({'xip':xip_full})
                        muted.update({'xim':xim_full})
                        muted.update({'npairs':mute[3]})
                        muted.update({'weight':mute[2]})
                        output_shear_shear_full.update({'{0}_{1}'.format(i,j):muted})
                    
        if self.params['lens_group'] != 'None':
            for i,j in self.all_calcs:
                if (i<self.lens_zbins)&(j<self.zbins)&(self.params['2pt_only'].lower() in [None,'pos-shear','all']):
                    
                    if (self.params['region_mode'] == 'pixellized') or (self.params['region_mode'] == 'both'):
                        gm = 6+2+6
                        shape = (gm, self.params['tbins'])
                        pairs_ring = [[np.zeros(shape) for ii in range(2)] for jk in range(self.jack_dict_tot['n_jck'])]
                        path = self.params['run_directory']+'/2pt/{0}_{1}_{2}/'.format(i,j,1)
                        for jci,jc in enumerate(np.unique(self.b)):
                            print("GGL: ", i,j,jc)
                            lla = '{0}'.format(jc)
                            dict_m = load_obj(path+lla)
                            pairsCC1 = dict_m['c1'][:gm]
                            pairsCC2 = dict_m['c2'][:gm]
                            pairs_auto = dict_m['a'][:gm]
                        
                            for prs in range(gm):
                     
                                pairsCC1[prs] += pairsCC2[prs]

                            pairs_ring[jci][1] = pairsCC1
                            pairs_ring[jci][0] = pairs_auto  
                    
                        pairs_ring = np.array(pairs_ring)
                        self.pairs_ring = pairs_ring
                        xi, xi_j, xir, xir_j,xi_im,xi_im_j,xi_imr,xi_imr_j,npairs,npairs_im,weight,weight_im,gammat_mr,gammat_mr_im,gammat_mr_j,gammat_mr_im_j,logr,rnom,meanr,meanlogr = collect(pairs_ring,self.jack_dict_tot['n_jck'],self.params['tbins'],'shear_pos',i,j)
                        
                        cov_xi = covariance_jck( xi_j.T,self.jack_dict_tot['n_jck'],'jackknife')
                        cov_xir = covariance_jck( xir_j.T,self.jack_dict_tot['n_jck'],'jackknife')
       
                        cov_xi_im = covariance_jck( xi_im_j.T,self.jack_dict_tot['n_jck'],'jackknife')
                        cov_xi_imr = covariance_jck( xi_imr_j.T,self.jack_dict_tot['n_jck'],'jackknife')
                        
                        cov_xi_mr = covariance_jck(xi_j.T-xir_j.T,self.jack_dict_tot['n_jck'],'jackknife')
                        cov_xi_mr_im = covariance_jck( xi_im_j.T-xi_imr_j.T,self.jack_dict_tot['n_jck'],'jackknife')
                    
                        files = dict()
                        files.update({'logr':logr})
                        files.update({'rnom':rnom})
                        files.update({'meanr':meanr})
                        files.update({'meanlogr':meanlogr})                   
                        
                        files.update({'gammat_compens':xi-xir})
                        files.update({'gammat_compens_jack':cov_xi_mr})
                        files.update({'gammat_compens_im':xi_im-xi_imr})
                        files.update({'gammat_compens_im_jack':cov_xi_mr_im})
                        
                        files.update({'gammat':xi})
                        files.update({'gammat_jack':xi_j})
                        files.update({'cov_gammat_jack':cov_xi})
                        files.update({'gammat_rndm':xir})
                        files.update({'gammat_rndm_jack':xir_j})
                        files.update({'cov_gammat_rndm_jack':cov_xir})
                    
                    
                        files.update({'gammat_im':xi_im})
                        files.update({'gammat_im_jack':xi_im_j})
                        files.update({'cov_gammat_im_jack':cov_xi_im})
                        files.update({'gammat_im_rndm':xi_imr})
                        files.update({'gammat_im_rndm_jack':xi_imr_j})
                        files.update({'cov_gammat_im_rndm_jack':cov_xi_imr})
                    
                        files.update({'npairs':npairs})
                        files.update({'npairs_rndm':npairs_im})
                        files.update({'weight':weight})
                        files.update({'weight_rndm':weight_im})
                        save_obj(path+'/results',files)
                        output_shear_pos.update({'{0}_{1}'.format(i,j):files})
                    
                    if (self.params['region_mode'] == 'full') or (self.params['region_mode'] == 'both'):
                        path = self.params['run_directory']+'/2pt/{0}_{1}_{2}/_full'.format(i,j,1)
                        if os.path.exists(path+'.pkl'):

                            
                            mute = load_obj(path)
                            
                            xi_full = (mute[0]/mute[2])
                            xir_full = (mute[1]/mute[3])
                            xi_im_full = (mute[4]/mute[2])
                            xi_imr_full = (mute[5]/mute[3])
          
                            muted= dict()
            
 
                
                            muted.update({'logr':mute[10]/mute[2]})
                            muted.update({'rnom':mute[11]/mute[2]})
                            muted.update({'meanr':mute[12]/mute[2]})
                            muted.update({'meanlogr':mute[13]/mute[2]})                   
                        
                            muted.update({'gammat_compens':mute[8]/mute[2]})
                    
                            muted.update({'gammat_compens_im':mute[9]/mute[2]})
    
                            muted.update({'gammat':xi_full})
                            muted.update({'gammat_rndm':xir_full})
                            muted.update({'gammat_im':xi_im_full})
                            muted.update({'gammat_im_rndm':xi_imr_full})
                            muted.update({'npairs':mute[6]})
                            muted.update({'npairs_rndm':mute[7]})
                            muted.update({'weight':mute[2]})
                            muted.update({'weight_rndm':mute[3]})
                            output_shear_pos_full.update({'{0}_{1}'.format(i,j):muted})
                    
                    
            for i,j in self.all_calcs:
                if (i<=j)&(j<self.lens_zbins)&(self.params['2pt_only'].lower() in [None,'pos-pos','all']):
                    
                    if (self.params['region_mode'] == 'pixellized') or (self.params['region_mode'] == 'both'):
                        gm = 12
                        shape = (gm, self.params['tbins'])
                        pairs_ring = [[np.zeros(shape) for ii in range(2)] for jk in range(self.jack_dict_tot['n_jck'])]
                        path = self.params['run_directory']+'/2pt/{0}_{1}_{2}/'.format(i,j,2)
                        for jci,jc in enumerate(np.unique(self.b)):
                            print("GG: ", i,j, jc)
                            lla = '{0}'.format(jc)
                            dict_m = load_obj(path+lla)
                            pairsCC1 = dict_m['c1'][:gm]
                            pairsCC2 = dict_m['c2'][:gm]
                            pairs_auto = dict_m['a'][:gm]
                            for prs in range(gm):
                                pairsCC1[prs] += pairsCC2[prs]

                            pairs_ring[jci][1] = pairsCC1
                            pairs_ring[jci][0] = pairs_auto  
                    
                        pairs_ring = np.array(pairs_ring)
                        self.pairs_ring = pairs_ring
                        xi, xi_j,npairs,weight,logr,rnom,meanr,meanlogr = collect(pairs_ring,self.jack_dict_tot['n_jck'],self.params['tbins'],'pos_pos',i,j)
                    
                        cov_xi = covariance_jck( xi_j.T,self.jack_dict_tot['n_jck'],'jackknife')
           
                        files = dict()
                        files.update({'logr':logr})
                        files.update({'rnom':rnom})
                        files.update({'meanr':meanr})
                        files.update({'meanlogr':meanlogr})  
                        files.update({'w':xi})
                        files.update({'w_jack':xi_j})
                        files.update({'cov_w_jack':cov_xi})
                        
                        files.update({'npairs':npairs})
                        files.update({'weight':weight})
                        save_obj(path+'/results',files)
                        output_pos_pos.update({'{0}_{1}'.format(i,j):files})
                        # compute_covariance & save results.

                    
                    if (self.params['region_mode'] == 'full') or (self.params['region_mode'] == 'both'):
                        path = self.params['run_directory']+'/2pt/{0}_{1}_{2}/_full'.format(i,j,2)
                        if os.path.exists(path+'.pkl'):
                            mute = load_obj(path)
                            xi =  (mute[0]-mute[1]-mute[2]+mute[3])/mute[3] 
                            muted = dict()
                            muted.update({'w':xi})
                            muted.update({'npairs':mute[4]})
                            muted.update({'weight':mute[0]})
                            muted.update({'logr':mute[8]/mute[0]})
                            muted.update({'rnom':mute[9]/mute[0]})
                            muted.update({'meanr':mute[10]/mute[0]})
                            muted.update({'meanlogr':mute[11]/mute[0]})    
                            output_pos_pos_full.update({'{0}_{1}'.format(i,j):muted})
                    
        self.output_shear_shear = output_shear_shear
        save_obj(self.params['run_directory']+'/2pt/shear_shear_pixellized',self.output_shear_shear)
        
        self.output_shear_pos = output_shear_pos
        save_obj(self.params['run_directory']+'/2pt/shear_pos_pixellized',self.output_shear_pos)
        
        self.output_pos_pos = output_pos_pos 
        save_obj(self.params['run_directory']+'/2pt/pos_pos_pixellized',self.output_pos_pos)
        
        # add full if present
        self.output_shear_shear_full = output_shear_shear_full
        save_obj(self.params['run_directory']+'/2pt/shear_shear_full',self.output_shear_shear_full)
        
        self.output_shear_pos_full = output_shear_pos_full
        save_obj(self.params['run_directory']+'/2pt/shear_pos_full',self.output_shear_pos_full)
        
        self.output_pos_pos_full = output_pos_pos_full 
        save_obj(self.params['run_directory']+'/2pt/pos_pos_full',self.output_pos_pos_full)
        
        
        
        return
