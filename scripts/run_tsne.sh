#!/bin/bash

log_path="data/tsne.log"
python scripts/tsne.py \
| tee $log_path