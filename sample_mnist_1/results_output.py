
# Create results directory if it doesn't exist
import glob
import logging
import os
import shutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Save results to results folder
results_dir = os.path.join(os.getcwd(), 'results')
snapshot_prefix = 'mnist_model'
    
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    logging.info("Created results directory: {}".format(results_dir))

# Find and move model snapshots and solverstates
model_files = glob.glob(snapshot_prefix + "_iter_*.caffemodel")
state_files = glob.glob(snapshot_prefix + "_iter_*.solverstate")

for file_path in model_files + state_files:
    if os.path.exists(file_path):
        dest_path = os.path.join(results_dir, os.path.basename(file_path))
        shutil.move(file_path, dest_path)
        logging.info("Moved: {} -> {}".format(file_path, dest_path))

# Copy lenet prototxt for each model file with matching filename
lenet_prototxt =  os.path.join(os.getcwd(), 'mnist_lenet.prototxt')ls 
if os.path.exists(lenet_prototxt):
    logging.info("Found lenet prototxt: {}".format(lenet_prototxt))
    for model_file in model_files:
        if os.path.exists(model_file):
            # Create prototxt copy with model's base name
            model_basename = os.path.basename(model_file)
            prototxt_name = os.path.splitext(model_basename)[0] + '.prototxt'
            dest_prototxt_path = os.path.join(results_dir, prototxt_name)
            shutil.copy2(lenet_prototxt, dest_prototxt_path)
            logging.info("Copied lenet prototxt: {} -> {}".format(lenet_prototxt, dest_prototxt_path))
else:
    logging.warning("Lenet prototxt not found: {}".format(lenet_prototxt))