from NGSIM_env.data.ngsim import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path", help="the path to the NGSIM csv file")
parser.add_argument("--scene", help="location", default='us-101')
args = parser.parse_args()

path = args.path
scene = args.scene
reader = ngsim_data(scene)
reader.read_from_csv(path)
reader.clean()
reader.dump(folder='NGSIM_env/data/processed/'+scene)
