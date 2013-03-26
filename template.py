#!/usr/bin/env python
''' Template that shows an example on how to use the grmEstimatorToolbox.
'''

# standard library
import os
import sys

# edit pythonpath
dir_ = os.path.realpath(__file__).replace('/example/template.py','')
sys.path.insert(0, dir_)

# project library
import grmToolbox

''' Simulation.
'''
grmToolbox.simulate()

''' Estimation.
'''
grmToolbox.estimate()

'''Evaluation
'''
grmToolbox.evaluate()
