gold_dict = {
    'objid'         : 'coadd_objects_id',
    'ra'            : 'ra',
    'dec'           : 'dec',
    'flags_gold'    : 'flags_gold',
    'flags_region'  : 'flags_badregion',
    'pzbin_col'     : None
    }

shape_dict = {
    'objid'         : 'coadd_objects_id',
    'e1'            : 'e1',
    'e2'            : 'e2',
    'w'             : 'weight',
    'm1'            : 'm1',
    'm2'            : 'm2',
    'c1'            : 'c1',
    'c2'            : 'c2',
    'flags_shape'   : 'flags',
    }

pz_dict = {
    'objid'         : 'coadd_objects_id',
    'pzbin'         : 'mean_z',
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
    'weight'        : None
    }

ran_dict = {
    'ra'            : 'ra',
    'dec'           : 'dec',
    'ranbincol'     : 'z'
    }    