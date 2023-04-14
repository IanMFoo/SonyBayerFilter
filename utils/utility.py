import sys
import os
import copy
import numpy as npf
import autograd.numpy as np
import logging
import traceback
import shelve
from datetime import datetime

from functools import reduce
import operator
import re

#*Argparse
argparse_universal_args = [
	["-f", "--filename", {'help':"Optimization Config file (YAML) path", 'required':True}],	#, default='test_config.yaml')
	["-m", "--mode", {'help':"Optimization or Evaluation mode", "choices":['opt','eval'], "default":'opt'}],
	["-o", "--override", {'help':"Override the default folder to save files out"}]
]

#*Logging
logging_filename = f'_t{int(round(datetime.now().timestamp()))}'
slurm_job_id = os.getenv('SLURM_JOB_ID')
if slurm_job_id is not None:
	logging_filename += f'_slurm-{slurm_job_id}'

loggingArgs = {'level':logging.INFO, 
			   'filename': 'logfile'+logging_filename+'.log',
			   'filemode': 'a',
			   'format': '%(asctime)s %(levelname)s|| %(message)s',
			   'datefmt':'%d-%b-%y %H:%M:%S'}
logging.basicConfig(**loggingArgs)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# Custom exception handler by re-assigning sys.excepthook() to logging module
def my_handler(type, value, tb):
	logging.getLogger().error("Uncaught exception {0}: {1}".format(str(type), str(value),
									exc_info=sys.exc_info()))
	logging.getLogger().error(traceback.format_tb(tb)[-1])
sys.excepthook = my_handler

logging.info('---Log: SONY Bayer Filter Adjoint Optimization---')
logging.info('Initializing Logger...')
logging.info(f'Current working directory: {os.getcwd()}')


#* HPC & Multithreading

def get_slurm_node_list( slurm_job_env_variable=None ):
	if slurm_job_env_variable is None:
		slurm_job_env_variable = os.getenv('SLURM_JOB_NODELIST')
	if slurm_job_env_variable is None:
		raise ValueError('Environment variable does not exist.')

	solo_node_pattern = r'hpc-\d\d-[\w]+'
	cluster_node_pattern = r'hpc-\d\d-\[.*?\]'
	solo_nodes = re.findall(solo_node_pattern, slurm_job_env_variable)
	cluster_nodes = re.findall(cluster_node_pattern, slurm_job_env_variable)
	inner_bracket_pattern = r'\[(.*?)\]'

	output_arr = solo_nodes
	for cluster_node in cluster_nodes:
		prefix = cluster_node.split('[')[0]
		inside_brackets = re.findall(inner_bracket_pattern, cluster_node)[0]
		# Split at commas and iterate through results
		for group in inside_brackets.split(','):
			# Split at hyphen. Get first and last number. Create string in range
			# from first to last.
			node_clump_split = group.split('-')
			starting_number = int(node_clump_split[0])
			try:
				ending_number = int(node_clump_split[1])
			except IndexError:
				ending_number = starting_number
			for i in range(starting_number, ending_number+1):
				# Can use print("{:02d}".format(1)) to turn a 1 into a '01'
				# string. 111 -> 111 still, in case nodes hit triple-digits.
				output_arr.append(prefix + "{:02d}".format(i))
	return output_arr


#* Other Functions


def isolate_filename(address):
	return os.path.basename(address)

def convert_root_folder(address, root_folder_new):
	new_address = os.path.join(root_folder_new, isolate_filename(address))
	return new_address

def generate_value_array(start,end,step):
    '''Returns a conveniently formatted output list according to np.arange() that can be copy-pasted to a JSON.'''
    output_array = json.dumps(np.arange(start,end+step,step).tolist())
    import pyperclip
    pyperclip.copy(output_array)
    return output_array

# Nested dictionary handling - https://stackoverflow.com/a/14692747
def get_by_path(root, items):
    """Access a nested object in root by item sequence."""
    return reduce(operator.getitem, items, root)

def set_by_path(root, items, value):
    """Set a value in a nested object in root by item sequence."""
    get_by_path(root, items[:-1])[items[-1]] = value

def del_by_path(root, items):
    """Delete a key-value in a nested object in root by item sequence."""
    del get_by_path(root, items[:-1])[items[-1]]
#######################################


def index_from_permittivity(permittivity_):
	'''Checks all permittivity values are real and then takes square root to give index.'''
	assert np.all(np.imag(permittivity_) == 0), 'Not expecting complex index values right now!'

	return np.sqrt(permittivity_)


def softplus( x_in, softplus_kappa ):
	'''Softplus is a smooth approximation to the ReLu function.
	It equals the sigmoid function when kappa = 1, and approaches the ReLu function as kappa -> infinity.
	See: https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html and https://pat.chormai.org/blog/2020-relu-softplus'''
	# return np.log( 1 + np.exp( x_in ) )
	return np.log( 1 + np.exp( softplus_kappa * x_in ) ) / softplus_kappa

def softplus_prime( x_in ):
	'''Derivative of the softplus function w.r.t. x, as defined in softplus()'''
	# return ( 1. / ( 1 + np.exp( -x_in ) ) )
	return ( 1. / ( 1 + np.exp( -softplus_kappa * x_in ) ) )