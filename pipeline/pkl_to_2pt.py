from astropy.io import fits
import numpy as np
import os
import pickle as p

root_dir = '2pt_pickles/complete_subsampled/'
    
def get_pkl_files(some_dir):
    filename_list = []
    for name in os.listdir(some_dir):
        if name[-3:]=='pkl':
            filename_list.append(name)
    return filename_list

all_pkl = get_pkl_files(root_dir)

print 'found these pkl files:',all_pkl

###################################################
# A dictionary for each correlation type

ss_bin_to_index = {'0_0':range(20),
                   '0_1':range(20,40),
                   '0_2':range(40,60),
                   '0_3':range(60,80),
                   '1_1':range(80,100),
                   '1_2':range(100,120),
                   '1_3':range(120,140),
                   '2_2':range(140,160),
                   '2_3':range(160,180),
                   '3_3':range(180,200)}
pp_bin_to_index = {'0_0':range(20),
                   '1_1':range(20,40),
                   '2_2':range(40,60),
                   '3_3':range(60,80),
                   '4_4':range(80,100)}
sp_bin_to_index = {'0_0':range(20),
                   '0_1':range(20,40),
                   '0_2':range(40,60),
                   '0_3':range(60,80),
                   '1_0':range(80,100),
                   '1_1':range(100,120),
                   '1_2':range(120,140),
                   '1_3':range(140,160),
                   '2_0':range(160,180),
                   '2_1':range(180,200),
                   '2_2':range(200,220),
                   '2_3':range(220,240),
                   '3_0':range(240,260),
                   '3_1':range(260,280),
                   '3_2':range(280,300),
                   '3_3':range(300,320),
                   '4_0':range(320,340),
                   '4_1':range(340,360),
                   '4_2':range(360,380),
                   '4_3':range(380,400)}


#################################################################################
# FULL 

if 'shear_shear_full.pkl' in all_pkl:
    print 'taking data from shear_shear_full'
    ssf = p.load(open(root_dir+'shear_shear_full.pkl', 'rb'))
    ang_xip_f, xip_f = np.zeros(200),np.zeros(200)
    ang_xim_f, xim_f = np.zeros(200),np.zeros(200)
    for i in ss_bin_to_index.keys():
        ang_xip_f[ss_bin_to_index[i]], xip_f[ss_bin_to_index[i]] = ssf[i]['meanr'],ssf[i]['xip'] 
        ang_xim_f[ss_bin_to_index[i]], xim_f[ss_bin_to_index[i]] = ssf[i]['meanr'],ssf[i]['xim']

if 'shear_pos_full.pkl' in all_pkl:
    print 'taking data from shear_pos_full'
    spf = p.load(open(root_dir+'shear_pos_full.pkl', 'rb'))
    ang_gammat_f, gammat_f = np.zeros(400),np.zeros(400)
    for i in sp_bin_to_index.keys():
        ang_gammat_f[sp_bin_to_index[i]], gammat_f[sp_bin_to_index[i]] = spf[i]['meanr'],spf[i]['gammat'] 
        
if 'pos_pos_full.pkl' in all_pkl:
    print 'taking data from pos_pos_full'
    ppf = p.load(open(root_dir+'pos_pos_full.pkl', 'rb'))
    ang_wtheta_f, wtheta_f = np.zeros(100),np.zeros(100)
    for i in pp_bin_to_index.keys():
        ang_wtheta_f[pp_bin_to_index[i]], wtheta_f[pp_bin_to_index[i]] = ppf[i]['meanr'],ppf[i]['w'] 


def create_fits_FULL(template_name,output_name):
    xip,header_xip = fits.getdata(template_name,'xip',header=True)
    xim,header_xim = fits.getdata(template_name,'xim',header=True)
    xip['value'] = xip_f
    xip['ang'] = ang_xip_f
    xim['value'] = xim_f
    xim['ang'] = ang_xim_f
    gammat,header_gammat = fits.getdata(template_name,'gammat',header=True)
    gammat['value'] = gammat_f
    gammat['ang'] = ang_gammat_f
    wtheta,header_wtheta = fits.getdata(template_name,'wtheta',header=True)
    wtheta['value'] = wtheta_f
    wtheta['ang'] = ang_wtheta_f
    #the sources and lenses should remain as the baseline?
    source,header_source = fits.getdata(template_name,'nz_source',header=True)
    #header_source['EXTNAME']='nz_source'
    lens,header_lens = fits.getdata(template_name,'nz_lens',header=True)
    t = fits.open(template_name)
    fits.append(output_name,t[0].data,t[0].header)
    fits.append(output_name,xip,header_xip)
    fits.append(output_name,xim,header_xim)
    fits.append(output_name,gammat,header_gammat)
    fits.append(output_name,wtheta,header_wtheta)
    fits.append(output_name,source,header_source)
    fits.append(output_name,lens,header_lens)
    t.close()
    return 0

create_fits_FULL('template.fits','mcalY3_full_subsampled.fits')
print 'Created full measurement datavector!'

#################################################################################

#################################################################################
# PIXELIZED

if 'shear_shear_pixellized.pkl' in all_pkl:
    print 'taking data from shear_shear_pixellized'
    ssp = p.load(open(root_dir+'shear_shear_pixellized.pkl', 'rb'))
    ang_xip_p, xip_p = np.zeros(200),np.zeros(200)
    ang_xim_p, xim_p = np.zeros(200),np.zeros(200)
    for i in ss_bin_to_index.keys():
        ang_xip_p[ss_bin_to_index[i]], xip_p[ss_bin_to_index[i]] = ssp[i]['meanr'],ssp[i]['xip'] 
        ang_xim_p[ss_bin_to_index[i]], xim_p[ss_bin_to_index[i]] = ssp[i]['meanr'],ssp[i]['xim']

if 'shear_pos_pixellized.pkl' in all_pkl:
    print 'taking data from shear_pos_pixellized'
    spp = p.load(open(root_dir+'shear_pos_pixellized.pkl', 'rb'))
    ang_gammat_p, gammat_p = np.zeros(400),np.zeros(400)
    for i in sp_bin_to_index.keys():
        ang_gammat_p[sp_bin_to_index[i]], gammat_p[sp_bin_to_index[i]] = spp[i]['meanr'],spp[i]['gammat'] 
        
if 'pos_pos_pixellized.pkl' in all_pkl:
    print 'taking data from pos_pos_pixellized'
    ppp = p.load(open(root_dir+'pos_pos_pixellized.pkl', 'rb'))
    ang_wtheta_p, wtheta_p = np.zeros(100),np.zeros(100)
    for i in pp_bin_to_index.keys():
        ang_wtheta_p[pp_bin_to_index[i]], wtheta_p[pp_bin_to_index[i]] = ppp[i]['meanr'],ppp[i]['w'] 


def create_fits_PIX(template_name,output_name):
    xip,header_xip = fits.getdata(template_name,'xip',header=True)
    xim,header_xim = fits.getdata(template_name,'xim',header=True)
    xip['value'] = xip_p
    xip['ang'] = ang_xip_p
    xim['value'] = xim_p
    xim['ang'] = ang_xim_p
    gammat,header_gammat = fits.getdata(template_name,'gammat',header=True)
    gammat['value'] = gammat_p
    gammat['ang'] = ang_gammat_p
    wtheta,header_wtheta = fits.getdata(template_name,'wtheta',header=True)
    wtheta['value'] = wtheta_p
    wtheta['ang'] = ang_wtheta_p
    #the sources and lenses should remain as the baseline?
    source,header_source = fits.getdata(template_name,'nz_source',header=True)
    #header_source['EXTNAME']='nz_source'
    lens,header_lens = fits.getdata(template_name,'nz_lens',header=True)
    t = fits.open(template_name)
    fits.append(output_name,t[0].data,t[0].header)
    fits.append(output_name,xip,header_xip)
    fits.append(output_name,xim,header_xim)
    fits.append(output_name,gammat,header_gammat)
    fits.append(output_name,wtheta,header_wtheta)
    fits.append(output_name,source,header_source)
    fits.append(output_name,lens,header_lens)
    t.close()
    return 0

create_fits_FULL('template.fits','mcalY3_pixellized_subsampled.fits')
print 'Created pixelized measurement datavector!'
#################################################################################

print 'All done!' 
