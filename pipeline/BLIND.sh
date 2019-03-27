py pkl_to_2pt.py ../y3_full/2pt mcalY3_pixellized_full.fits

python blind_2pt_usingcosmosis.py  -i blinding_params_template.ini -b add -u mcalY3_pixellized_full.fits
