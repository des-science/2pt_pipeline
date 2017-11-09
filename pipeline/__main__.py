from .pipeline import launch_pipeline, run_stage, stages
import argparse


parser = argparse.ArgumentParser(description="Run the multi-probe pipeline")
parser.add_argument("parameter_file", help="YAML configuration file")
parser.add_argument("--stage", default="", choices=stages, help="YAML configuration file")
parser.add_argument("--manual", action="store_true", help="YAML configuration file")
parser.add_argument("--mpi", action='store_true', help="Run under MPI (only if --stage=stage also set)")

args = parser.parse_args()

if args.stage:
    run_stage(args.parameter_file, args.stage, args.mpi)
else:
    launch_pipeline(args.parameter_file, args.manual)

