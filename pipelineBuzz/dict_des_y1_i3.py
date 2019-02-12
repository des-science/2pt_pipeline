gold_dict = {
    'objid'         : 'coadd_objects_id',
    'ra'            : None,
    'dec'           : None,
    'flags_gold'    : None,
    'flags_region'  : None,
    'pzbin_col'     : None
    }

shape_dict = {
    'objid'         : 'coadd_objects_id',
    'e1'            : 'e1',
    'e2'            : 'e2',
    'weight'        : 'weight',
    'm1'            : 'm',
    'm2'            : 'm',
    'c1'            : 'c1',
    'c2'            : 'c2',
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

pz_stack_dict = {
    'objid'         : 'coadd_objects_id',
    'pzstack'       : 'z_mc',
    'pzflags'       : None,
    'pzw'           : None
    }

lens_pz_dict = {
    'objid'         : 'COADD_OBJECTS_ID',
    'pzbin'         : 'ZREDMAGIC',
    'pzstack'       : 'ZREDMAGIC',
    'pzerr'         : 'ZREDMAGIC_E',
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
    'ranbincol'     : 'Z'
    }    
