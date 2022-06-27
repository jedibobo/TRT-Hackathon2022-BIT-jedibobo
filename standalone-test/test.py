import imp
from clifs import CLIFS
import clip
import os 

os.environ['INPUT_DIR']='data/input'
os.environ['OUTPUT_DIR']='data/output'
#os.environ['INPUT_DIR']='/home/lyb/code/clifs/data/input'
#os.environ['OUTPUT_DIR']='/home/lyb/code/clifs/data/output'
os.environ['MODEL']='ViT-B/32' # RN50 RN101 RN50x4 RN50x16 RN50x64 ViT-B/32 ViT-B/16 ViT-L/14 ViT-L/14@336px

clifs = CLIFS()

query = "A BMW car"
query_results = clifs.search(query)
