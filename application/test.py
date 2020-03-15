#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys,os
import argparse
lib_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(lib_path, '../'))

from configs.configs import cfg

if __name__ == "__main__":
    p = argparse.ArgumentParser(description = 'app test for ai algorithms demos')
    p.add_argument('-place_holder',default= 'default test', help = 'one or more intergers is need')
    args = p.parse_args()
    #print(args.place_holder, cfg.data.linear_regression)
    linear_regression_cmd = 'python3 samples/ml/regression/linear/sgd.py'
    os.system(linear_regression_cmd)