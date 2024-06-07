import datasets
from datasets import load_dataset
from PIL import Image
from pathlib import Path
import pandas as pd
import os
import json
import tqdm
import argparse
import shutil
import numpy as np

np.random.seed(0)

"""
Changes proportion of examples in metadata_sampled.json
"""
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_dir", type=str, default="~/.cache")
  parser.add_argument("--output_dir", type=str, default="./vqa_examples")
  args = parser.parse_args()

  dataset_prop = {
      "DocVQA": 500,
      "ChartQA": 500,
      "NewYorker": 1000,
      "WikiArt": 500,
      "TextVQA": 500,
  }

  dataset_json = []
  for dataset_name in dataset_prop.keys():
    with open(f"{args.output_dir}/{dataset_name}/data.json") as f:
      data = json.load(f)
      dataset_json.extend(np.random.choice(data, dataset_prop[dataset_name]))

  with open(f"{args.output_dir}/metadata_sampled.json", "w") as f:
    json.dump(dataset_json, f, indent=4)