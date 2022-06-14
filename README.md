# CONVIQT: Contrastive Video Quality Estimator

**Pavan C. Madhusudana**, Neil Birkbeck, Yilin Wang, Balu Adsumilli and Alan C. Bovik

This is the official repository of the paper [CONVIQT: Contrastive Video Quality Estimator](https://arxiv.org/abs/2110.13266)

## Usage
The code has been tested on Linux systems with python 3.7. Please refer to [requirements.txt](requirements.txt) for installing dependent packages.

### Running CONVIQT
In order to obtain quality score, checkpoints needs to be downloaded. The following command can be used to download the checkpoint.
```
wget -L https://utexas.box.com/shared/static/rhpa8nkcfzpvdguo97n2d5dbn4qb03z8.tar -O models/CONTRIQUE_checkpoint25.tar -q --show-progress
wget -L https://utexas.box.com/shared/static/7s8348b0imqe27qkgq8lojfc2od1631a.tar -O models/CONVIQT_checkpoint10.tar -q --show-progress
```
Alternatively, the checkpoints can also be downloaded using these links [link1](https://utexas.box.com/s/rhpa8nkcfzpvdguo97n2d5dbn4qb03z8) and [link2](https://utexas.box.com/s/7s8348b0imqe27qkgq8lojfc2od1631a)

Google drive link for the checkpoints [link1](https://drive.google.com/file/d/1pmaomNVFhDgPSREgHBzZSu-SuGzNJyEt/view?usp=drive_web) and [link2](https://drive.google.com/file/d/1f3h8gha8YbuLTngzAkmf7MB79rYcywin/view?usp=sharing)

### Obtaining Quality Scores
We provide trained regressor models in [models](models) directory which can be used for predicting image quality using features obtained from CONVIQT model. For demonstration purposes, some sample videos provided in the [sample_videos](sample_videos) folder.

For blind quality prediction, the following commands can be used.
```
python3 demo_score.py --video_path sample_videos/30.mp4 --spatial_model_path models/CONTRIQUE_checkpoint25.tar --temporal_model_path models/CONVIQT_checkpoint10.tar --linear_regressor_path models/YouTube_UGC.save
python3 demo_score.py --video_path sample_videos/Flips_crf_48_30fps.webm --spatial_model_path models/CONTRIQUE_checkpoint25.tar --temporal_model_path models/CONVIQT_checkpoint10.tar --linear_regressor_path models/LIVE_YT_HFR.save
```

### Obtaining CONVIQT Features
For calculating CONVIQT features, the following commands can be used. The features are saved in '.npy' format.
```
python3 demo_feat.py --video_path sample_videos/30.mp4 --spatial_model_path models/CONTRIQUE_checkpoint25.tar --temporal_model_path models/CONVIQT_checkpoint10.tar --features_save_path features.npy
python3 demo_feat.py --video_path sample_videos/Flips_crf_48_30fps.webm --spatial_model_path models/CONTRIQUE_checkpoint25.tar --temporal_model_path models/CONVIQT_checkpoint10.tar --features_save_path features.npy
```

## Training CONVIQT
### Download Training Data
Create a directory ```mkdir training_data``` to store videos used for training CONVIQT. Run the following commands to download and unzip training data containing videos with synthetic distortions.
```
bash data_download.sh
bash data_unzip.sh
```

For UGC videos download Kinetics dataset [link](https://www.deepmind.com/open-source/kinetics) and unzip the data. For training CONVIQT only directories parts_0-parts_10 present in Kinetics-400 dataset are needed.

### Training Model
Download csv files containing path to videos and corresponding distortion classes.
```

wget -L https://utexas.box.com/shared/static/63pvroz3287j1kj7ja0gv81vw2txnwlw.csv -O csv_files/file_names_ugc.csv -q --show-progress
wget -L https://utexas.box.com/shared/static/migniec2yb07vc8kz002kub658840dun.csv -O csv_files/file_names_syn.csv -q --show-progress
```
The above files can also be downloaded manually using these links [link1](https://utexas.box.com/s/63pvroz3287j1kj7ja0gv81vw2txnwlw), [link2](https://utexas.box.com/s/migniec2yb07vc8kz002kub658840dun)
Google drive links [link1](https://drive.google.com/file/d/1N7EGdS-mobbcWmJUOvblCyQ_xWQ4hwjc/view?usp=sharing), [link2](https://drive.google.com/file/d/109w9c6t8EAEP_yrLKJtzsYAB1-QbLXoy/view?usp=sharing)

# Spatial Feature Extraction
Spatial features are extracted using [CONTRIQUE](https://github.com/pavancm/CONTRIQUE) model using the following commands
```
python3 spatial_feature_extract_syn.py
python3 spatial_feature_extract_ugc.py
```
Extracted spatial features are saved in training_data directory.

For training CONVIQT with a single GPU the following command can be used
```
python3 train.py --batch_size 256 --lr 0.6 --epochs 25
```

### Training Linear Regressor
After CONVIQT model training is complete, a linear regressor is trained using CONVIQT features and corresponding ground truth quality scores using the following command.

```
python3 train_regressor.py --feat_path feat.npy --ground_truth_path scores.npy --alpha 0.1
```

## Contact
Please contact Pavan (pavan.madhusudana@gmail.com) if you have any questions, suggestions or corrections to the above implementation.

## Acknowledgement
This repository is built on the [CONTRIQUE](https://github.com/pavancm/CONTRIQUE) repository.
