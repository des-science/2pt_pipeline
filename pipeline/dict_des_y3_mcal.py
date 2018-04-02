gold_dict = {
    'objid'         : 'coadd_objects_id',
    'ra'            : None,
    'dec'           : None,
    'flags_gold'    : None,
    'flags_region'  : None,
    'pzbin_col'     : None
    }

shape_dict = {
    'objid'         : 'coadd_object_id',
    'e1'            : 'e1',
    'e2'            : 'e2',
    'm1'            : 'R11',
    'm2'            : 'R22',
    'cov00'         : 'covmat_0_0',
    'cov11'         : 'covmat_1_1',
    'snr'           : 'snr',
    'ra'            : 'ra',
    'dec'           : 'dec',
    'flags'         : 'flags_select',
    }

pz_bin_dict = {
    'objid'         : 'coadd_objects_id',
    'pzbin'         : 'mean_z',
    'pzflags'       : None,
    'pzw'           : None
    }

bpz_dict = {
    'objid' : 'coadd_object_id',
    'pzbin' : 'bpz_zmean_sof', #this will only work for 'catalog/bpz/unsheared'
    'pzstack' : 'bpz_zmc_sof'
    #'pz_1p' : 'bpz_zmean_sof_1p',
    #'pz_1m' : 'bpz_zmean_sof_1m',
    #'pz_2p' : 'bpz_zmean_sof_2p',
    #'pz_2m' : 'bpz_zmean_sof_2m'
}

dnf_dict = {
    'objid' : 'coadd_object_id',
    'pzbin' : 'dnf_zmean_sof' #this will only work for 'catalog/dnf/unsheared'
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
    'ra'            : 'RA',
    'dec'           : 'DEC',
    'weight'        : 'weight'
    }

ran_dict = {
    'ra'            : 'RA',
    'dec'           : 'DEC',
    'ranbincol'     : 'z'
    }    

index_dict = {
    'u':0,
    '1p':1,
    '1m':2,
    '2p':3,
    '2m':4
}
