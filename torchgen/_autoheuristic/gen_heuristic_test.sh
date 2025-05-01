#!/bin/bash

dir=""
data="/home/zhajia/pytorch/torchgen/_autoheuristic/mm/gpu_profile_results.txt"
python train_decision_mem.py ${data} --heuristic-name GPURankingDemo --save-dot
