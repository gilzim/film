#!/bin/bash

log_path="data/run_model_fibn.log"
python scripts/run_model.py \
--program_generator data/cbn_layer3_batch96_dropout20.pt \
--execution_engine data/cbn_layer3_batch96_dropout20.pt \
--input_question_h5 data/val_questions.h5 \
--input_features_h5 data/val_features.h5 \
--num_samples 3000 \
--batch_size 100 \
--output_program_stats_dir img/stats/FiBN \
| tee $log_path