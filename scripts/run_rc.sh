#!/bin/bash
python model.py \
--train_file ./data/training/format/train_format.json \
--valid_file ./data/training/format/valid_format.json \
--BERT_DIR ./models/bert/
--cuda_device 0 \
--epochs 20 \
--batch_size 12 \
--learning_rate 5e-5 \
--training \
--subj_info \
--obj_info \
--lock_layers \
--ugc_info \
--global_info \
--pair_info \
--gate_mechanism \
--ugc_attention_type 1
