#!/usr/bin/env bash

TRAIN_CSV=data/trainLabels.csv

POINT1=data/points1.json
POINT2=data/points2.json
SLOT=data/slot.json

SAMPLE_SUB=data/sample_submission.csv

NAME=hl_pred_train
EXP_DIR=hl_pred_train

PARAMS=hl_fit/training/epoch_500.3c

python whales/main.py --test-csv-url ${SAMPLE_SUB} --name ${NAME} \
--test-dir-url ${IMG_DIR} --train-dir-url ${IMG_DIR} \
--train-csv-url ${TRAIN_CSV} --glr 0.01 --mb-size 64 --crop-h 224 --crop-w 224 \
--method momentum --arch gscp_smaller --monitor-freq 100 --n-samples-valid 1 --loss-freq 5 --do-pca 1 --pca-scale 0.01 \
--fc-l2-reg 0.05 --conv-l2-reg 0.0005 --do-mean 1 --aug-params crop1_buckets --glr-burnout 15 --glr-decay 0.9955 \
--valid-seed 7300 --slot-annotations-url ${SLOT} --show-images 30 --valid-freq 1 \
--process-recipe-name fetch_rob_crop_recipe --point1-annotations-url ${POINT1} \
--point2-annotations-url ${POINT2} --buckets 60 --target-name crop1 --mode crop1 \
--exp-dir-url ${EXP_DIR} --real-valid-shuffle --valid-partial-batches \
--load-params-url ${PARAMS} --global-saver-url global \
--gen-crop1-train --no-train --n-samples-test 1