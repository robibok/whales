#!/usr/bin/env bash

TRAIN_CSV=data/trainLabels.csv

AUTO_INDYGO=data/indygo_annos_all.json
RYJ_CONN=data/ryj_conn_anno.json
NEW_CONN=data/new_conn.csv
POINT1=data/points1.json
POINT2=data/points2.json

SAMPLE_SUB=data/sample_submission.csv

NAME=net3_fit_step2
EXP_DIR=net3_fit_step2

PARAMS=net3_fit_step1/training/epoch_140.3c

python whales/main.py --train-dir-url ${IMG_DIR} --train-csv-url ${TRAIN_CSV} \
--glr 0.0001 --mb-size 12 --crop-h 512 --crop-w 512 --method momentum --monitor-freq 100 --n-samples-valid 5 \
--loss-freq 5 --do-pca 1 --pca-scale 0.01 --fc-l2-reg 0.005 --conv-l2-reg 0.0005 --do-mean 1 --glr-decay 0.9955 \
--n-fc 256 --n-first 32 --valid-seed 7300 --n-classes 447 --process-recipe-name fetch_example_anno_indygo \
--auto-indygo-annotations-url ${AUTO_INDYGO} \
--ryj-conn-annotations-url ${RYJ_CONN} --nof-best-crops -1 --global-saver-url global \
--point1-annotations-url ${POINT1} --point2-annotations-url ${POINT2} \
--new-conn-csv-url ${NEW_CONN} --show-images 10 \
--equalize --name ${NAME} --margin 40 --arch new_gsc3_for_512 \
--aug-params magik_z --indygo-equalize --glr-burnout 0 --valid-partial-batches \
--train-pool-size 1 --test-pool-size 1 --train-part 1.0 \
--load-params-url ${PARAMS} \
--target-name final --mode final \
--exp-dir-url ${EXP_DIR} --n-epochs 61