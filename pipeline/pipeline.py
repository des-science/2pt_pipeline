import yaml
import numpy as np
from .compute_covariance import ComputeCovariance
from .measure_2pt import Measure2Point
from .nofz import nofz 
from .text_2pt import Text2Point
from .cosmology import ParameterEstimation
from .submit import submit_hybrid, submit_mpi, submit_single, get_cores_per_node
from .write_fits import WriteFits

stages = {
    "nofz":nofz,
    "2pt":Measure2Point,
    "cov":ComputeCovariance,
    "write_fits":WriteFits,
    "fits2txt":Text2Point,
    "cosmo":ParameterEstimation,
}


def run_stage(param_file, stage_name, mpi):
    if mpi:
        print "mpi mode"
        import mpi4py.MPI
        comm = mpi4py.MPI.COMM_WORLD
    else:
        comm = None

    stage_class = stages[stage_name]
    stage_class.execute(param_file,comm)


def launch_pipeline(param_file, manual_submission):
    do_submit = not manual_submission

    #Run the serial stage
    nofz.execute(param_file, None)

    #We need more cores if we have more bins, so load this now.
    params = yaml.load(open(param_file))
    run_dir = params['run_directory'] + "/"
    nbin = max(params['zbins'], len(params['lens_zbins'])-1)

    #Submit the parallel stages
    #The 2pt measurement
    nodes = 10
    cmd = "python -m pipeline {} --stage 2pt --mpi".format(param_file)
    jobid_2pt = submit_hybrid(nodes, "debug", "00:30:00", "2pt", "2pt.sub", cmd, do_submit)

    #The covariance matrix calculation
    #blocks = np.loadtxt(ComputeCovariance.outputs['blocks'])
    #nodes = np.ceil(blocks / get_cores_per_node()).astype(int)
    nodes = 10
    cmd = "python -m pipeline {} --stage cov --mpi".format(param_file)
    jobid_cov = submit_mpi(nodes, "regular", "03:30:00", "cov", "cov.sub", cmd, do_submit)

    # These two stages can be combined into a single job
    cmd = """
    python -m pipeline {} --stage write_fits
    python -m pipeline {} --stage fits2txt
    """.format(param_file, param_file)
    jobid_collate = submit_single("00:05:00", "fits2txt", "fits2txt.sub", cmd, do_submit, previous_jobs=[jobid_2pt,jobid_cov])
  
    cmd = "python -m pipeline {} --stage cosmo --mpi".format(param_file)
    print cmd
    submit_mpi(1, "regular", "2:00:00", "cosmo", "cosmo.sub", cmd, do_submit, previous_jobs=[jobid_collate])

