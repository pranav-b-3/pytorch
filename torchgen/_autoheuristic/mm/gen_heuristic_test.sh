#!/bin/bash

dir=""
data="gpu_profile_results.txt"
python train_decision_mm.py ${data} --heuristic-name GPURankingDemo --save-dot
