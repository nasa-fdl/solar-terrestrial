#!/bin/bash -v

gunzip $PWD/omni_min2016.asc.gz
for h in m180 m120 m60 0 60 120 180 240 300 360 420 480 540 600 660 720 ; do ipython kp_regress.py $h; done
