#!/bin/bash
set -e

python3 setup.py develop
python3 raisimGymTorch/env/envs/rsg_a1/tester.py -w  $1 $2 --episodes 10
