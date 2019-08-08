# 2pt_pipeline
A collection of tools and modules for producing the 2pt data products and necessary components for parameter estimation using them.

# Use

To run the pipeline interactively with the supplied Y1 yaml file, use (e.g., from 2pt_pipeline/) python -m pipeline --stage stage_name mcal.yaml

stage_name is one of nofz, 2pt, cov, write_fits.

# External components (probably incomplete)

 - https://github.com/joezuntz/2point/
 - https://github.com/esheldon/fitsio
 - https://github.com/rmjarvis/TreeCorr/ (to calculate 2pt functions, stage 2pt)
 - https://bitbucket.org/timeifler/cosmolike (to calculate covariance, stage cov)
 - https://github.com/des-science/destest/ (For Y3 improvements, using data source/selector classes there.)

# How to use the docker/shifter image at NERSC

Using shifter at NERSC will significantly decrease load times for python code
using many cores. It will also eliminate annoying dependencies at NERSC :). To use it follow the instructions below.

First, run:

`shifterimg -v pull docker:jderose/2pt_pipeline:latest`

This loads the 2pt_pipeline docker image to NERSC. You can then run jobs using
this image by specifying this image in your submission script with the SLURM argument:

--image=docker:jderose/2pt_pipeline

e.g. in an interactive job:

`salloc -N 1 -A des -q interactive -t 4:00:00 --image=docker:jderose/2pt_pipeline`

or in a normal batch submission script:
```
&#35SBATCH -p regular
&#35SBATCH -A des
&#35SBATCH -t 36:00:00
&#35SBATCH -N 1
&#35SBATCH -C haswell
&#35SBATCH -L SCRATCH
&#35SBATCH --image=docker:jderose/2pt_pipeline

srun -n 32 shifter python3 -m /2pt_pipeline/pipeline --stage 2pt <config>
```

You will also likely need to specify a filesystem to mount to the docker image. For example, extending on the above SLURM script:

```
&#35SBATCH -p regular
&#35SBATCH -A des
&#35SBATCH -t 36:00:00
&#35SBATCH -N 1
&#35SBATCH -C haswell
&#35SBATCH -L SCRATCH
&#35SBATCH --image=docker:jderose/2pt_pipeline
&#35SBATCH --volume="/global/cscratch1/sd/<username>/output/:/output;/global/cscratch1/sd/<username>/input/:/input"

srun -n 32 shifter python3 -m /2pt_pipeline/pipeline --stage 2pt <config>

```

will allow you to access the directories /global/cscratch1/sd/<username>/output/ and /global/cscratch1/sd/<username>/input/ as /output/ and /input/ when running, so make sure to replace the relevant directories in your config file.

For much more information about docker/shifter at NERSC, see here: https://docs.nersc.gov/programming/shifter/how-to-use/
