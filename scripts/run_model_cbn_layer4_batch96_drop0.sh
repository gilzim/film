#!/bin/bash

python scripts/run_model.py \
--program_generator data/cbn_layer4_batch96_dropout0.pt \
--execution_engine data/cbn_layer4_batch96_dropout0.pt \
--input_question_h5 data/train_questions.h5 \
--input_features_h5 data/train_features.h5