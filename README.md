# fy2015-replication
This repository contains the complete configuration files necessary for the replication of Fei and Yeung (2015), "Temporal Models for Predicting Student Dropout in Massive Open Online Courses" using the MOOC Replication Framework (MORF). The complete results of this replication are described in Gardner, Yang, Baker, and Brooks (2018), "Enabling End-To-End Machine Learning Replicability: A Case Study in Educational Data Mining."

## Guide to the contents of this repo:

`docker`: contains dockerfile and necessary scripts to build the docker image. This image can also be pulled directly from docker cloud by running `docker pull themorf/morf-public:fy2015-replication`.

`config`: contains two subdirectories, `holdout` and `cv`, with configuration files to reproduce the experiment using the holdout and cross-validation architectures, respectively. Note that weeks are zero-indexed (so `week_0` actually uses one week of features, and `week_4` uses weeks one through five, utilizing the method described in the original Fei and Yeung paper).

## Executing the experiments described in this repo:

To execute one of the trials described here (where a trial is a specific model evaluated with features up to a specific week number), use the MORF API functions:

```
from morf.utils.submit import easy_submit
easy_submit(TODO)
```

Note that the complete extraction-training-testing pipeline may take several hours. Also note that if you are using a job which utilizes `fork_features()`, the job it is forking from must be executed first.


