import h5py as h
import numpy as np
import treecorr
import os

subsampled = True
bslop = 0.5
PROCESS = int(os.environ['SLURM_PROCID']) #sbatching a job on 10 nodes will spawn 10 tasks (if you do it right) 
THREADS = int(os.environ['OMP_NUM_THREADS']) #must be exported in the batch job script, 32 for cori/haswell and 68 for cori/knl
outpath = '/global/homes/s/seccolf/des-science/my_2pt_pipeline/measurement/outputs/subsampled_bslop0.1/'



######################################
if subsampled:
    print '\nRunning on SUBSAMPLED Y3 catalog!!\n'
    catname = '/global/cscratch1/sd/troxel/cats_des_y3/Y3_mastercat_v2_6_20_18_subsampled.h5'
else:
    print '\nRunning on FULL Y3 catalog!!\n'
    catname = '/global/cscratch1/sd/troxel/cats_des_y3/Y3_mastercat_v2_6_20_18.h5'
#######
# get masks
f = h.File(catname,'r')
metacal_mask = np.array(f['index/select'])
mask_1p = np.array(f['index/select_1p'])
mask_1m = np.array(f['index/select_1m'])
mask_2p = np.array(f['index/select_2p'])
mask_2m = np.array(f['index/select_2m'])


###########                                                                                                             
#split by zbins                                                                                                        
# use /global/cscratch1/sd/seccolf/y3_fullcat/nofz/nz_source_zbin.h5 for the full catalog 
if subsampled:
    zname = '/global/homes/s/seccolf/des-science/my_2pt_pipeline/measurement/nz_source_zbin.h5'
else:
    zname = '/global/cscratch1/sd/seccolf/y3_fullcat/nofz/nz_source_zbin.h5'

n = h.File(zname,'r')#this hdf5 file contains the result from the previous step that computes the n(z)                                                                            
zbin_array = np.array(n['nofz/zbin'])

ind1 = np.where( zbin_array==0 )[0]
ind2 = np.where( zbin_array==1 )[0]
ind3 = np.where( zbin_array==2 )[0]
ind4 = np.where( zbin_array==3 )[0]


################################################################
#get catalog properties 
dgamma = 2*0.01
ra = np.array(f['catalog/gold/ra'])[metacal_mask]
dec = np.array(f['catalog/gold/dec'])[metacal_mask]
#ra = np.array(f['catalog/metacal/unsheared/ra'])[metacal_mask]
#dec = np.array(f['catalog/metacal/unsheared/dec'])[metacal_mask]
e1 = np.array(f['catalog/metacal/unsheared/e_1'])[metacal_mask]
e2 = np.array(f['catalog/metacal/unsheared/e_2'])[metacal_mask]
R11 = np.array(f['catalog/metacal/unsheared/R11'])[metacal_mask]
R22 = np.array(f['catalog/metacal/unsheared/R22'])[metacal_mask]

################################################################
#defining quantities for zbin 1, and subtracting mean shear:    

ind1_1p = np.where( np.array(n['nofz/zbin_1p'])==0 )
ind1_1m = np.where( np.array(n['nofz/zbin_1m'])==0 )
ind1_2p = np.where( np.array(n['nofz/zbin_2p'])==0 )
ind1_2m = np.where( np.array(n['nofz/zbin_2m'])==0 )

R11s_1 = ( np.array(f['catalog/metacal/unsheared/e_1'])[mask_1p][ind1_1p].mean() - 
           np.array(f['catalog/metacal/unsheared/e_1'])[mask_1m][ind1_1m].mean() )/dgamma
R22s_1 = ( np.array(f['catalog/metacal/unsheared/e_2'])[mask_2p][ind1_2p].mean() - 
           np.array(f['catalog/metacal/unsheared/e_2'])[mask_2m][ind1_2m].mean() )/dgamma           
R11_1 = np.mean(R11[ind1])                                                           
R22_1 = np.mean(R22[ind1])                                                           

R11tot_1 = R11s_1 + R11_1
R22tot_1 = R22s_1 + R22_1

print 'In zbin1: <R11>=%f, <R22>=%f, <R11s>=%f, <R22s>=%f'%(R11_1,R22_1,R11s_1,R22s_1)

g1_1 = (e1[ind1] - np.mean(e1[ind1])) / R11tot_1
g2_1 = (e2[ind1] - np.mean(e2[ind1])) / R22tot_1
ra_1 = ra[ind1]
dec_1 = dec[ind1]

#######################
#same for zbin 2

ind2_1p = np.where( np.array(n['nofz/zbin_1p'])==1 )
ind2_1m = np.where( np.array(n['nofz/zbin_1m'])==1 )
ind2_2p = np.where( np.array(n['nofz/zbin_2p'])==1 )
ind2_2m = np.where( np.array(n['nofz/zbin_2m'])==1 )

R11s_2 = ( np.array(f['catalog/metacal/unsheared/e_1'])[mask_1p][ind2_1p].mean() -
           np.array(f['catalog/metacal/unsheared/e_1'])[mask_1m][ind2_1m].mean() )/dgamma
R22s_2 = ( np.array(f['catalog/metacal/unsheared/e_2'])[mask_2p][ind2_2p].mean() -
           np.array(f['catalog/metacal/unsheared/e_2'])[mask_2m][ind2_2m].mean() )/dgamma
R11_2 = np.mean(R11[ind2])
R22_2 = np.mean(R22[ind2])

R11tot_2 = R11s_2 + R11_2
R22tot_2 = R22s_2 + R22_2

print 'In zbin2: <R11>=%f, <R22>=%f, <R11s>=%f, <R22s>=%f'%(R11_2,R22_2,R11s_2,R22s_2)

g1_2 = (e1[ind2] - np.mean(e1[ind2])) / R11tot_2
g2_2 = (e2[ind2] - np.mean(e2[ind2])) / R22tot_2
ra_2 = ra[ind2]
dec_2 = dec[ind2]
#######################                                                    
#same for zbin 3   
ind3_1p = np.where( np.array(n['nofz/zbin_1p'])==2 )
ind3_1m = np.where( np.array(n['nofz/zbin_1m'])==2 )
ind3_2p = np.where( np.array(n['nofz/zbin_2p'])==2 )
ind3_2m = np.where( np.array(n['nofz/zbin_2m'])==2 )

R11s_3 = ( np.array(f['catalog/metacal/unsheared/e_1'])[mask_1p][ind3_1p].mean() -
           np.array(f['catalog/metacal/unsheared/e_1'])[mask_1m][ind3_1m].mean() )/dgamma
R22s_3 = ( np.array(f['catalog/metacal/unsheared/e_2'])[mask_2p][ind3_2p].mean() -
           np.array(f['catalog/metacal/unsheared/e_2'])[mask_2m][ind3_2m].mean() )/dgamma
R11_3 = np.mean(R11[ind3])
R22_3 = np.mean(R22[ind3])

R11tot_3 = R11s_3 + R11_3
R22tot_3 = R22s_3 + R22_3

print 'In zbin3: <R11>=%f, <R22>=%f, <R11s>=%f, <R22s>=%f'%(R11_3,R22_3,R11s_3,R22s_3)

g1_3 = (e1[ind3] - np.mean(e1[ind3])) / R11tot_3
g2_3 = (e2[ind3] - np.mean(e2[ind3])) / R22tot_3
ra_3 = ra[ind3]
dec_3 = dec[ind3]
#######################                                                        
#same for zbin 4
ind4_1p = np.where( np.array(n['nofz/zbin_1p'])==3 )
ind4_1m = np.where( np.array(n['nofz/zbin_1m'])==3 )
ind4_2p = np.where( np.array(n['nofz/zbin_2p'])==3 )
ind4_2m = np.where( np.array(n['nofz/zbin_2m'])==3 )

R11s_4 = ( np.array(f['catalog/metacal/unsheared/e_1'])[mask_1p][ind4_1p].mean() -
           np.array(f['catalog/metacal/unsheared/e_1'])[mask_1m][ind4_1m].mean() )/dgamma
R22s_4 = ( np.array(f['catalog/metacal/unsheared/e_2'])[mask_2p][ind4_2p].mean() -
           np.array(f['catalog/metacal/unsheared/e_2'])[mask_2m][ind4_2m].mean() )/dgamma
R11_4 = np.mean(R11[ind4])
R22_4 = np.mean(R22[ind4])

R11tot_4 = R11s_4 + R11_4
R22tot_4 = R22s_4 + R22_4

print 'In zbin4: <R11>=%f, <R22>=%f, <R11s>=%f, <R22s>=%f'%(R11_4,R22_4,R11s_4,R22s_4)

g1_4 = (e1[ind4] - np.mean(e1[ind4])) / R11tot_4
g2_4 = (e2[ind4] - np.mean(e2[ind4])) / R22tot_4
ra_4 = ra[ind4]
dec_4 = dec[ind4]

#######################################################
# setting up 2pt measurement

#dictionary to set up parallel execution:
gamma1 = {1:g1_1 , 2:g1_2 , 3:g1_3 , 4:g1_4}
gamma2 = {1:g2_1 , 2:g2_2 , 3:g2_3 , 4:g2_4}
RA = {1:ra_1 , 2:ra_2 , 3:ra_3 , 4:ra_4}
DEC = {1:dec_1 , 2:dec_2 , 3:dec_3 , 4:dec_4}
#now gammai[j] represents the i-th shear component of the j-th redshift bin

#for PROCESS in np.arange(10):

if PROCESS==0: 
    i,j = (1,1)
if PROCESS==1: 
    i,j = (1,2)
if PROCESS==2: 
    i,j = (1,3)
if PROCESS==3:
    i,j = (1,4)
if PROCESS==4:
    i,j = (2,2)
if PROCESS==5:
    i,j = (2,3)
if PROCESS==6:
    i,j = (2,4)
if PROCESS==7:
    i,j = (3,3)
if PROCESS==8:
    i,j = (3,4)
if PROCESS==9:
    i,j = (4,4)

print '\ntreecorr cat length: zbin%d=%d, zbin%d=%d'%(i,len(gamma1[i]),j,len(gamma1[j]))
cat1 = treecorr.Catalog(g1=gamma1[i],g2=gamma2[i],ra=RA[i],dec=DEC[i],ra_units='deg',dec_units='deg')
cat2 = treecorr.Catalog(g1=gamma1[j],g2=gamma2[j],ra=RA[j],dec=DEC[j],ra_units='deg',dec_units='deg')

print 'ID for this job: ',PROCESS,'\nDoing combination i=',i, 'j=',j,'\nUsing',THREADS,'cores'
GG = treecorr.GGCorrelation(nbins=20,min_sep=2.5,max_sep=250.0,sep_units='arcmin',verbose=2,bin_slop=bslop)
GG.process(cat1,cat2,num_threads=THREADS)
GG.write(outpath+'tomo_test_bslop'+str(bslop)+'_'+str(i)+str(j)+'.txt')


