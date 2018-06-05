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
| holdout | 0 | lr  | 1275035 |temp|
| holdout | 0 | rnn | 1275045 | temp |
| holdout | 0 | svm | 1275193 | temp |
| holdout | 0 | lstm | 1275041 | temp|
| holdout | 1 | lr | 1275049 | temp |
| holdout | 1 | lstm | 1275055 | temp |
| holdout | 1 | rnn | 1275059 | temp |
| holdout | 1 | svm | 1275197 | temp |
| holdout | 2 | lr | 1275063 | temp |
| holdout | 2 | lstm | 1275071 | temp |
| holdout | 2 | rnn | 1275074 | temp |
| holdout | 2 | svm | 1275201 | temp |
| holdout | 3 | lr | 1275077 | temp |
| holdout | 3 | rnn | 1275081 | temp |
| holdout | 3 | lstm | 1275083 | temp |
| holdout | 3 | svm | 1275203 | temp |
| holdout | 4 | lr | 1275331 | temp |
| holdout | 4 | rnn | 1275335 | temp |
| holdout | 4 | lstm | 1275339 | temp |
| holdout | 4 | svm | 1275341 | temp |
| cv | 0 | lr | 1275087 | temp |
| cv | 0 | rnn | 1275091 | temp |
| cv | 0 | lstm | 1275095 | temp |
| cv | 0 | svm | 1275207 | temp |
| cv | 1 | lr | 1275101 | temp |
| cv | 1 | rnn | 1275103 | temp |
| cv | 1 | lstm | 1275107 | temp |
| cv | 1 | svm | 1275211 | temp |
| cv | 2 | lr | 1275113 | temp |
| cv | 2 | rnn | 1275119 | temp |
| cv | 2 | lstm | 1275121 | temp |
| cv | 2 | svm | 1275213 | temp |
| cv | 3 | lr | 1275129 | temp |
| cv | 3 | rnn | 1275133 | temp |
| cv | 3 | lstm | 1275135 | temp |
| cv | 3 | svm | 1275215 | temp |
| cv | 4 | lr | 1275345 | temp |
| cv | 4 | rnn | 1275347 | temp |
| cv | 4 | lstm | 1275351 |temp |
| cv | 4 | svm | 1275355 | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1275355.svg)](https://doi.org/10.5281/zenodo.1275355) |



