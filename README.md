# 2pt_pipeline
A collection of tools and modules for producing the 2pt data products and necessary components for parameter estimation using them.

See discussion here:
https://docs.google.com/document/d/12qjP--RHMTClAv1tOVSctNz0jJOm9Jp_p_7RCvzUv7o/edit?usp=sharing

# Use

To run the pipeline interactively with the supplied Y1 yaml file, use (e.g., from 2pt_pipeline/) python -m pipeline --stage stage_name mcal.yaml

stage_name is one of nofz, 2pt, cov, write_fits. 

# External components (probably incomplete)

 - https://github.com/joezuntz/2point/
 - https://github.com/esheldon/fitsio
 - https://github.com/rmjarvis/TreeCorr/ (to calculate 2pt functions, stage 2pt)
 - https://bitbucket.org/timeifler/cosmolike (to calculate covariance, stage cov)
 - https://github.com/des-science/destest/ (For Y3 improvements, using data source/selector classes there.)
