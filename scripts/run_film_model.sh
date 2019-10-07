#!/bin/bash

python scripts/run_model.py \
--program_generator data/film.pt \
--execution_engine data/film.pt \
--input_question_h5 data/train_questions.h5 \
--input_features_h5 data/train_features.h5