#!/bin/bash
export DEMODEL=lcdm
echo $1
python pipeline/blind_2pt_usingcosmosis.py  -i pipeline/blinding_params_redmagic.ini -b add -u ${1}
rm $1
