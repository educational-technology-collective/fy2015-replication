# fy2015-replication
This repository contains the complete configuration files necessary for the replication of Fei and Yeung (2015), "Temporal Models for Predicting Student Dropout in Massive Open Online Courses" using the MOOC Replication Framework (MORF). The complete results of this replication are described in Gardner, Yang, Baker, and Brooks (2018), "Enabling End-To-End Machine Learning Replicability: A Case Study in Educational Data Mining."

## Guide to the contents of this repo:

`docker`: contains dockerfile and necessary scripts to build the docker image. This image can also be pulled directly from docker cloud by running `docker pull themorf/morf-public:fy2015-replication`.

`config`: contains two subdirectories, `holdout` and `cv`, with configuration files to reproduce the experiment using the holdout and cross-validation architectures, respectively. Note that weeks are zero-indexed (so `week_0` actually uses one week of features, and `week_4` uses weeks one through five, utilizing the method described in the original Fei and Yeung paper).

## Executing the experiments described in this repo:

To execute one of the trials described here (where a trial is a specific model evaluated with features up to a specific week number), use the MORF API functions:

```
from morf.utils.submit import easy_submit
easy_submit(client_config_url="https://raw.githubusercontent.com/educational-technology-collective/fy2015-replication/master/config/holdout/week_4/svm/controller.py", email_to="your-email@example.com")
```

Note that the complete extraction-training-testing pipeline may take several hours. Also note that if you are using a job which utilizes `fork_features()`, the job it is forking from must be executed first.


| Experiment | Week | Model | Zenodo Deposition ID | DOI | 
| ------------- | ------------- | ------------- | ------------- |
| holdout | 0 | lr  | 1275035 | |
| holdout | 0 | rnn | 1275045 |  |
| holdout | 0 | svm | 1275193 |  |
| holdout | 0 | lstm | 1275041 | |
| holdout | 1 | lr | 1275049 |  |
| holdout | 1 | lstm | 1275055 |  |
| holdout | 1 | rnn | 1275059 |  |
| holdout | 1 | svm | 1275197 |  |
| holdout | 2 | lr | 1275063 |  |
| holdout | 2 | lstm | 1275071 |  |
| holdout | 2 | rnn | 1275074 |  |
| holdout | 2 | svm | 1275201 |  |
| holdout | 3 | lr | 1275077 |  |
| holdout | 3 | rnn | 1275081 |  |
| holdout | 3 | lstm | 1275083 |  |
| holdout | 3 | svm | 1275203 |  |
| holdout | 4 | lr | 1275331 |  |
| holdout | 4 | rnn | 1275335 |  |
| holdout | 4 | lstm | 1275339 |  |
| holdout | 4 | svm | 1275341 |  |
| cv | 0 | lr | 1275087 |  |
| cv | 0 | rnn | 1275091 |  |
| cv | 0 | lstm | 1275095 |  |
| cv | 0 | svm | 1275207 |  |
| cv | 1 | lr | 1275101 |  |
| cv | 1 | rnn | 1275103 |  |
| cv | 1 | lstm | 1275107 |  |
| cv | 1 | svm | 1275211 |  |
| cv | 2 | lr | 1275113 |  |
| cv | 2 | rnn | 1275119 |  |
| cv | 2 | lstm | 1275121 |  |
| cv | 2 | svm | 1275213 |  |
| cv | 3 | lr | 1275129 |  |
| cv | 3 | rnn | 1275133 |  |
| cv | 3 | lstm | 1275135 |  |
| cv | 3 | svm | 1275215 |  |
| cv | 4 | lr | 1275345 |  |
| cv | 4 | rnn | 1275347 |  |
| cv | 4 | lstm | 1275351 | |
| cv | 4 | svm | 1275355 | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1275355.svg)](https://doi.org/10.5281/zenodo.1275355) |



