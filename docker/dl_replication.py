# Copyright (c) 2018 The Regents of the University of Michigan
# and the University of Pennsylvania
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
A script to extract features in replication of Mi and Yeung (2015).
"""

import argparse
import subprocess
import os
from feature_extraction.feature_extractor import main as extract_features
from feature_extraction.sql_utils import extract_coursera_sql_data
from modeling.modeling_utils import aggregate_session_input_data, fetch_test_features_path, fetch_trained_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="execute feature extraction, training, or testing.")
    parser.add_argument('--course', required=True, help="an s3 pointer to a course")
    parser.add_argument('--session', required=False, help="3-digit course run number")
    parser.add_argument('--mode', required=True, help="mode to run image in; {extract, train, test}")
    parser.add_argument("--n_weeks", required=True, type = int, help = "number of weeks to extract features from")
    parser.add_argument("--model_type", required=False, help="modeling algorithm to use; needed for training and testing")
    parser.add_argument("--fold_num", required=False, help="fold number for cross-validation")
    args = parser.parse_args()
    if args.mode == 'extract':
        # setup the mysql database
        extract_coursera_sql_data(args.course, args.session)
        extract_features(course_name = args.course, run_number = args.session, n_weeks = args.n_weeks)
    if args.mode == 'train':
        # call train_lstm.py script here with --mode = train --model_type = some model type
        # file names for features and labels; this follows MORF file naming format
        assert args.n_weeks >= 0
        assert args.model_type
        feature_file = aggregate_session_input_data(file_type = "features", course = args.course)
        label_file = aggregate_session_input_data(file_type = "labels", course = args.course)
        output_file = "/output/{}_{}_model".format(args.course, args.session)
        cmd = "python3 modeling/train_lstm.py --feature_data {} --label_data {} --output_file {} --n_feature 7 --mode train --model_type {}".format(feature_file, label_file, output_file, args.model_type)
        subprocess.call(cmd, shell = True)
    if args.mode == 'test':
        assert args.n_weeks >= 0
        assert args.model_type
        feature_file = fetch_test_features_path()
        model_file = fetch_trained_model_path()
        output_file = "/output/{}_test.csv".format(args.course)
        cmd = "python3 modeling/train_lstm.py --feature_data {} --trained_model {} --output_file {} --n_feature 7 --mode test --model_type {}".format(feature_file, model_file, output_file, args.model_type)
        subprocess.call(cmd, shell=True)
    if args.mode == "cv":
        # train the model, and save the model to input dir (because it is still used as input for the next step and won't be saved by MORF)
        assert args.fold_num
        train_feature_file = os.path.join("input", args.course, "_".join([args.course, args.fold_num, "train", "features.csv"]))
        train_label_file = os.path.join("input", args.course, "_".join([args.course, args.fold_num, "train", "labels.csv"]))
        model_file = "/input/{}_model".format(args.course)
        train_cmd = "python3 modeling/train_lstm.py --feature_data {} --label_data {} --output_file {} --n_feature 7 --mode train --model_type {}".format(
            train_feature_file, train_label_file, model_file, args.model_type)
        subprocess.call(train_cmd, shell=True)
        test_feature_file = os.path.join("input", args.course, "_".join([args.course, args.fold_num, "test", "features.csv"]))
        test_output_file = "/output/{}_{}_test.csv".format(args.course, args.fold_num)
        test_cmd = "python3 modeling/train_lstm.py --feature_data {} --trained_model {} --output_file {} --n_feature 7 --mode test --model_type {}".format(test_feature_file, model_file, test_output_file, args.model_type)
        subprocess.call(test_cmd, shell=True)
