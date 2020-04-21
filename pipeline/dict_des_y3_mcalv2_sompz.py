regions_dict = {
   'region' : 'region'}

gold_dict = {
    'objid'         : 'coadd_objects_id',
    'ra'            : 'ra',
    'dec'           : 'dec',
    'flags_gold'    : 'flags_gold',
    'flags_region'  : None,
    'pzbin_col'     : None,
    'hpix'          : 'hpix_16384'
}

shape_dict = {
    'objid'         : 'coadd_object_id',
    'e1'            : 'e_1',
    'e2'            : 'e_2',
    'm1'            : 'R11',
    'm2'            : 'R22',
    'cov00'         : 'covmat_1_1',
    'cov11'         : 'covmat_2_2',
    'snr'           : 'snr',
    'ra'            : 'ra',
    'dec'           : 'dec',
    'flags'         : 'flags_select',
    'weight'        : 'weight'
    }

pz_bin_dict = {
    'objid'         : 'coadd_objects_id',
    'pzbin'         : 'mean_z',
    'pzflags'       : None,
    'pzw'           : None
    }

pz_dict = {
    'objid' : 'coadd_object_id',
    'pzbin' : 'bhat', #this will only work for 'catalog/bpz/unsheared'
    'pzstack' : 'cell_wide'
    #'pz_1p' : 'bpz_zmean_sof_1p',
    #'pz_1m' : 'bpz_zmean_sof_1m',
    #'pz_2p' : 'bpz_zmean_sof_2p',
    #'pz_2m' : 'bpz_zmean_sof_2m'
}

pz_stack_dict = {
    'objid'         : 'coadd_objects_id',
    'pzstack'       : 'z_mc',
    'pzflags'       : None,
    'pzw'           : None
    }

lens_pz_dict = {
    'objid'         : 'coadd_object_id',
    'pzbin'         : 'zredmagic',
    'pzstack'       : 'zredmagic',
    'pzerr'         : 'zredmagic_e',
    'weight'        : None
    }

lens_dict = {
    'objid'         : 'COADD_OBJECTS_ID',
    'ra'            : 'ra',
    'dec'           : 'dec',
    'weight'        : 'weight'
    }

ran_dict = {
    'ra'            : 'ra',
    'dec'           : 'dec',
    'ranbincol'     : 'z'
    }    

index_dict = {
    'u':0,
    '1p':1,
    '1m':2,
    '2p':3,
    '2m':4
}
