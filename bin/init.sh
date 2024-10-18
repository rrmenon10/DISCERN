#!/bin/bash

set -exu

conda env create -f env.yaml
mkdir -p hf_cache