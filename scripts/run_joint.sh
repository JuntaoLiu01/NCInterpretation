#!/bin/bash
python joint.py \
--rc_train_file ./data/training/format/train_format.json \
--rc_valid_file ./data/training/format/valid_format.json \
--para_train_file ./data/explanation/train_format.json \
--para_valid_file ./data/explanation/valid_format.json \
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
