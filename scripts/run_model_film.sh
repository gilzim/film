#!/bin/bash

log_path="data/run_model_film.log"
python scripts/run_model.py \
--program_generator data/film.pt \
--execution_engine data/film.pt \
--input_question_h5 data/val_questions.h5 \
--input_features_h5 data/val_features.h5 \
--num_samples 3000 \
--batch_size 100 \
--output_program_stats_dir img/stats/FiLM \
| tee $log_path