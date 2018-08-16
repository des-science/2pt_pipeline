#source /global/homes/s/seccolf/cosmosis/LOAD_STUFF

source path_to_your_setup-cosmosis-nersc

python pipeline/blind_2pt_usingcosmosis.py -s NEWTEST -i pipeline/blinding_params_template.ini -b add -u pipeline/pegY3cosmicshear_unblinded.fits
