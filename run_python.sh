#!/bin/bash
#casia
# configs
#CONFIG="casia.base.casia_arcface"
#CONFIG="casia.base.casia_cosface"
#CONFIG="casia.base.casia_sphereface"
# CONFIG="casia.base.casia_expface"

# ms1mv3
# CONFIG="ms1mv3.base.ms1mv3_arcface"
# CONFIG="ms1mv3.base.ms1mv3_cosface"
# CONFIG="ms1mv3.base.ms1mv3_sphereface"
# CONFIG="ms1mv3.base.ms1mv3_expface"
# CONFIG="ms1mv3.base.ms1mv3_expface_uniface"
# CONFIG="ms1mv3.base.ms1mv3_magface"
CONFIG="ms1mv3.base.ms1mv3_naiveface"

conda run --live-stream --name base python train.py --config $CONFIG



#test
conda run --live-stream --name base python test/eval_ijbc.py --config $CONFIG
conda run --live-stream --name base python test/eval_ijbb.py --config $CONFIG
conda run --live-stream --name base python test/eval_veri.py --config $CONFIG
