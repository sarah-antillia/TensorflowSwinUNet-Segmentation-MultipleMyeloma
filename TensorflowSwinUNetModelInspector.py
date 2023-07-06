#
# TensorflowSwinUNetModelInspector.py
#
import os
import sys
import traceback
from ConfigParser import ConfigParser

from TensorflowSwinUNet import TensorflowSwinUNet
MODEL = "model"

if __name__ == "__main__":

  try:
    # Default config_file
    config_file    = "./train_eval_infer.config"
    # You can specify config_file on your command line parammeter.
    if len(sys.argv) == 2:
      confi_file= sys.argv[1]
      if not os.path.exists(config_file):
         raise Exception("Not found " + config_file)
     
    config   = ConfigParser(config_file)
    
    width    = config.get(MODEL, "image_width")
    height   = config.get(MODEL, "image_height")

    if not (width == height and  height % 128 == 0 and width % 128 == 0):
      raise Exception("Image width should be a multiple of 128. For example 128, 256, 512")
    
    # Create a UNetMolde and compile
    model    = TensorflowSwinUNet(config_file)
    model.inspect("./model.png")
    # Please download and install graphviz for your OS

  except:
    traceback.print_exc()
    

