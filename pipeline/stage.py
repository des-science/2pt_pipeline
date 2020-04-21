from __future__ import print_function, division
import time
import os
import errno

TWO_POINT_NAMES = ['xip', 'xim', 'gammat', 'wtheta']
NOFZ_NAMES = ['nz_source', 'nz_lens']
COV_NAME = 'COVMAT'

class PipelineStage(object):
    name = "error"
    inputs = {}
    outputs = {}

    def __init__(self, param_file):
        import yaml
        self.params = yaml.load(open(param_file))
        self.params['param_file'] = param_file
        self.base_dir = self.params['run_directory']
        self.output_dir = os.path.join(self.base_dir, self.name)
        if not os.path.exists(self.output_dir):
            mkdir(self.output_dir)

    @classmethod
    def execute(cls, param_file, comm):
        name = cls.__name__
        print("Preparing stage {}".format(name))
        stage = cls(param_file)
        stage.comm = comm
        t0 = time.time()
        print("Running stage {}".format(name))
        stage.run()
        stage.write()
        t1 = time.time()
        print("Done stage {} in {} seconds".format(name, t1-t0))

    def input_path(self, name):
        section, filename = self.inputs[name]
        return os.path.join(self.base_dir, section, filename)

    def output_path(self, name):
        filename = self.outputs[name]
        return os.path.join(self.output_dir, filename)



def mkdir(path):
    #This is much nicer in python 3.
    try:
        os.makedirs(path)
    except OSError as error:
        if error.errno == errno.EEXIST:
            if os.path.isdir(path):
                #error is that dir already exists; fine - no error
                pass
            else:
                #error is that file with name of dir exists already
                raise ValueError("Tried to create dir %s but file with name exists already"%path)
        elif error.errno == errno.ENOTDIR:
            #some part of the path (not the end) already exists as a file
            raise ValueError("Tried to create dir %s but some part of the path already exists as a file"%path)
        else:
            #Some other kind of error making directory
            raise
