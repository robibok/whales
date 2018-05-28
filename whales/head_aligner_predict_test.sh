#!/usr/bin/env bash

TRAIN_CSV=data/trainLabels.csv

POINT1=data/points1.json
POINT2=data/points2.json
SLOT=data/slot.json

RYJ_CONN=data/ryj_conn_anno.json

# replace path if you want to use the output of head_localizer_predict scripts
AUTO_SLOT=data/all_bbox_test.json

SAMPLE_SUB=data/sample_submission.csv

NAME=ha_predict_test
EXP_DIR=ha_predict_test

PARAMS=ha_fit/training/epoch_500.3c

python whales/main.py --train-dir-url ${IMG_DIR} --train-csv-url ${TRAIN_CSV} \
--glr 0.0005 --mb-size 24 --crop-h 256 --crop-w 256 --method momentum --monitor-freq 100 --n-samples-valid 5 \
--loss-freq 5 --do-pca 1 --pca-scale 0.01 --fc-l2-reg 0.05 --conv-l2-reg 0.0005 --do-mean 1 --aug-params whales_4 \
--glr-decay 0.9955 --n-fc 256 --n-first 32 --valid-seed 7300 --n-classes 447 --process-recipe-name fetch_example_crop2 \
--auto-slot-annotations-url ${AUTO_SLOT} \
--ryj-conn-annotations-url ${RYJ_CONN} --nof-best-crops -1 \
--point1-annotations-url ${POINT1} --point2-annotations-url ${POINT2} \
--show-images 10 --equalize --name ${NAME} --arch new_gsc --mode crop2 --target-name crop2 \
--valid-freq 5 --buckets 60 --exp-dir-url ${EXP_DIR} \
--n-samples-test 20 \
--load-params-url ${PARAMS} --global-saver-url global \
--gen-crop2-test --no-train --test-csv-url ${SAMPLE_SUB} --test-dir-url ${IMG_DIR}