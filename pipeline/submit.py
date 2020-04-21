from __future__ import print_function, division
import subprocess
import os


def get_cores_per_node():
    nersc_host = os.environ.get("NERSC_HOST", "None")
    if nersc_host == "edison":
        cores_per_node = 24
    elif nersc_host == "cori":
        cores_per_node = 32
    elif nersc_host == "None":
        cores_per_node = 4
        print("Guessing {} cores per node".format(cores_per_node))
    else:
        raise ValueError("Unknown number of cores on this machine - fix in submit.py")
    return cores_per_node

#assumes full set of cores per node
mpi_template = """#!/bin/bash -l
#SBATCH -p {queue}
#SBATCH -N {nodes}
#SBATCH -t {time}
#SBATCH -J {name}
#SBATCH -o {name}.log
#SBATCH -C haswell
{dependency}

cd $SLURM_SUBMIT_DIR
export OMP_NUM_THREADS=1
srun -n {cores} {command}
"""



#assumes one process per node
hybrid_template = """#!/bin/bash -l
#SBATCH -p {queue}
#SBATCH -N {nodes}
#SBATCH -t {time}
#SBATCH -J {name}
#SBATCH -o {name}.log
#SBATCH -C haswell
{dependency}

cd $SLURM_SUBMIT_DIR
export OMP_NUM_THREADS={cores_per_node}
srun -n {nodes} -c {cores_per_node} {command}
"""


#assumes single core in shared queue
single_template = """#!/bin/bash -l
#SBATCH -p debug
#SBATCH -n 1
#SBATCH -t {time}
#SBATCH -J {name}
#SBATCH -o {name}.log
#SBATCH -C haswell
{dependency}

cd $SLURM_SUBMIT_DIR
{command}

"""

def make_deps(previous_jobs):
    if previous_jobs:
        if any(p is None for p in previous_jobs):
            dependency = ""
        else:
            dependency = "#SBATCH -d afterok:{0}".format(":".join(previous_jobs))
    else:
        dependency = ""
    return dependency

def submit(script, script_file, do_submit):
    f = open(script_file, 'w')
    f.write(script)
    f.close()
    cmd = "sbatch {}".format(script_file)
    if not do_submit:
        print(cmd)
        return None

    #will raise error on failure of submission
    output = subprocess.check_output(cmd, shell=True)
    if not output.startswith("Submitted batch job"):
        raise RuntimeError("Submission failed: {}".format(output))
    job_id = output.split()[3]
    return job_id

def submit_mpi(nodes, queue, time, name, script_file, command, do_submit, previous_jobs=None):
    dependency = make_deps(previous_jobs)
    print("Assuming 32 cores per node - modify for edison")
    cores = nodes*32
    job_script = mpi_template.format(**locals())
    return submit(job_script, script_file, do_submit)


def submit_hybrid(nodes, queue, time, name, script_file, command, do_submit, previous_jobs=None):
    #edison value
    cores_per_node = get_cores_per_node()
    print("Assuming 32 cores per node - modify for edison")
    dependency = make_deps(previous_jobs)
    job_script = hybrid_template.format(**locals())
    return submit(job_script, script_file, do_submit)


def submit_single(time, name, script_file, command, do_submit, previous_jobs=None):
    dependency = make_deps(previous_jobs)
    job_script = single_template.format(**locals())
    return submit(job_script, script_file, do_submit)
