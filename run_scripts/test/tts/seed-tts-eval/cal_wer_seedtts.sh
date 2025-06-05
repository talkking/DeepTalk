#!/bin/bash
PATH=$1
LANG=$2
CUDA_VISIBLE_DEVICES=4 /usr/bin/python run_wer.py $PATH/wav_text.txt $PATH/wer.txt $LANG
