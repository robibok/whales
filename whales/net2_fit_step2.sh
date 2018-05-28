#!/usr/bin/env bash

TRAIN_CSV=data/trainLabels.csv

AUTO_INDYGO=data/indygo_annos_all.json
RYJ_CONN=data/ryj_conn_anno.json
NEW_CONN=data/new_conn.csv
POINT1=data/points1.json
POINT2=data/points2.json

SAMPLE_SUB=data/sample_submission.csv

NAME=net2_fit_step2
EXP_DIR=net2_fit_step2

PARAMS=net2_fit_step1/training/epoch_500.3c

python whales/main.py --train-dir-url ${IMG_DIR} --train-csv-url ${TRAIN_CSV} \
--glr 0.0002 --mb-size 24 --crop-h 256 --crop-w 256 --method momentum --monitor-freq 100 --n-samples-valid 5 \
--loss-freq 5 --do-pca 1 --pca-scale 0.01 --fc-l2-reg 0.05 --conv-l2-reg 0.0005 --do-mean 1 --glr-decay 0.955 \
--n-fc 256 --n-first 32 --valid-seed 7300 --n-classes 447 --process-recipe-name fetch_example_anno_indygo \
--auto-indygo-annotations-url ${AUTO_INDYGO} \
--ryj-conn-annotations-url ${RYJ_CONN} --nof-best-crops -1 \
--point1-annotations-url ${POINT1} --point2-annotations-url ${POINT2} \
--new-conn-csv-url ${NEW_CONN} --show-images 10 --global-saver-url global \
--equalize --name ${NAME} --margin 40 --real-valid-shuffle \
--arch new_gsc3 --aug-params magik_z --indygo-equalize --glr-burnout 0 --valid-partial-batches --train-part 1.0 \
--train-pool-size 5 --exp-dir-url ${EXP_DIR} --target-name final --mode final --train-part 1.0 --n-epochs 101 \
--load-params-url ${PARAMS}