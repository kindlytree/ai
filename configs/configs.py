#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys,os
from easydict import EasyDict as edict

lib_path = os.path.abspath(".")
lib_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(lib_path)

def load_cfg_from_file(filename):
    import yaml
    with open(filename, 'r') as f:
        yml_cfg = edict(yaml.load(f))
        return yml_cfg

cfg = load_cfg_from_file(os.path.join(lib_path, 'configs.yml'))

def merge_cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg,  cfg)

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v
