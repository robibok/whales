# Winning solution for the Right Whale Recognition competition on Kaggle

A detailed description of the approach can be found on <http://deepsense.io/deep-learning-right-whale-recognition-kaggle>.

## Requirements
- a bunch of Python libraries (try `pip install -r requirements.txt`)
- Nvidia GPU and cuDNN v2 (or newer)

## Additional training data
- manual annotations for bounding boxes: `data/all_bbox.json`
- manual annotations for bonnet-tip and blowhead: `data/points1.json` and `data/points2.json`
- manual annotations for continuous/broken classification of callosity patterns: `data/new_conn.csv` (and its older and poorer version `data/old_conn.json`)

## Reproducing the results

As reproducing everything from scratch is very time consuming, we include results of head localization and head alignment steps, as well as weights for some of the models.

### Precomputed annotations
- `all_bbox.json` and `all_bbox_test.json` - results of head localization
- `indygo_annos_all.json` - results of head alignment

### Trained models
- `models/head_localizer.3c` - a sample model for head localization
- `models/head_aligner.3c` - a sample model for head alignment
- `models/model1_params.3c` - one of the classifiers used in the final submission
- `models/model2_params.3c` - one of the classifiers used in the final submission
- `models/model3_params.3c` - one of the classifiers used in the final submission

### Setting everything up

1. Set path to original images in `init.sh`
2. (Optional - not needed if you only care about using final classifiers) Use `convert_25.sh input_path output_path` to produce downsized images, set path in `init.sh`.
3. Run `source init.sh`

The scripts print tons of debug info to stdout, you may prefer to redirect their output.

### Generating the submission file (from dumped models)

```
./whales/model1_predict.sh
./whales/model2_predict.sh
./whales/model3_predict.sh

python whales/blender.py model1_predict/submit1.csv model2_predict/submit1.csv model3_predict/submit1.csv submission.csv
```

The score on private LB should be around 0.6.

### Training the models on your own

Note that the classifiers were trained on the provided annotations. This means that if you train a new head localizer/aligner, compute new json files with annotations, and feed it to one of the provided models, it may achieve worse results. To avoid this, either retrain the model on new data or train it from scratch.

Depending on the GPU used, each step will take several hours (and training all final classifiers even more).

#### Head localizer

```
./whales/head_localizer_fit.sh
./whales/head_localizer_predict_train.sh
./whales/head_localizer_predict_test.sh
```

This will create json files with coordinates of bounding boxes: `hl_pred_test/test_bbox.json` and `hl_pred_train/train_bbox.json`.

#### Head aligner

```
./whales/head_aligner_fit.sh
./whales/head_aligner_predict_train.sh
./whales/head_aligner_predict_test.sh
```

This will create json files with coordinates of bonnet-tip and blowhead: `ha_predict_test/test_indygo.json` and `ha_predict_train/train_indygo.json`.

#### Final classifier

During the contest we were tampering with the learning rate manually. The scripts try to mimick this, but may not work that well.

```
./whales/net1_fit_step1.sh
./whales/net1_fit_step2.sh

./whales/net2_fit_step1.sh
./whales/net2_fit_step2.sh

./whales/net3_fit_step1.sh
./whales/net3_fit_step2.sh
./whales/net3_fit_step3.sh

./whales/net1_predict.sh
./whales/net2_predict.sh
./whales/net3_predict.sh
```

#### Blending

```
python whales/blender.py net1_predict/submit1.csv net2_predict/submit1.csv net3_predict/submit1.csv submission.csv
```










