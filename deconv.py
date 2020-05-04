import argparse
from deconv_algorithm import DeconvAlgorithm

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="config.json path", action="store")
parser.add_argument("--image", help="image file path", action="store")
args = parser.parse_args()
deconv = DeconvAlgorithm(args)