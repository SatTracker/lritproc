#!/bin/bash

IN="./LRIT Files"
OUT="./LRIT Output"

python lritproc.py -av --debug ${IN} ${OUT} --mkdir
