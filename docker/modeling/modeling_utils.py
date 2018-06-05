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
Utility functions for replication of Mi and Yeung (2015).
"""

import pandas as pd
import os


def aggregate_session_input_data(file_type, course, input_dir = "/input"):
    """
    Aggregate all csv data files matching pattern within input_dir (recursive file search), and write to a single file in input_dir.
    :param type: {"labels" or "features"}.
    :param dest_dir:
    :return:
    """
    valid_types = ("features", "labels")
    course_dir = os.path.join(input_dir, course)
    if file_type not in valid_types:
        print("[ERROR] specify either features or labels as type.")
        return None
    else:
        df_out = pd.DataFrame()
        for root, dirs, files in os.walk(course_dir, topdown=False):
            for session in dirs:
                session_csv = "{}_{}_{}.csv".format(course, session, file_type)
                session_feats = os.path.join(root, session, session_csv)
                print(session_feats)
                session_df = pd.read_csv(session_feats)
                df_out = pd.concat([df_out, session_df])
        # write single csv file
        outfile = "{}_{}.csv".format(course, file_type)
        outpath = os.path.join(input_dir, outfile)
        df_out.to_csv(outpath, index = False)
    return outpath

def fetch_test_features_path(dir = "/input"):
    """
    Fetch path to features csv.
    :param dir: dir to search.
    :return: path to features csv (string).
    """
    feature_file = None
    for root, dirs, files in os.walk(dir):
        for file in files:
            p = os.path.join(root, file)
            if p.endswith("features.csv"):
                feature_file = p
    return feature_file


def fetch_trained_model_path(input_dir = "/input"):
    # find files inside dir ending with "_model"
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            p = os.path.join(root, file)
            print(p)
            if p.endswith("_model") and not p.startswith("."):
                model_file = p
                print("[INFO] loading model file: {}".format(model_file))
    return model_file

