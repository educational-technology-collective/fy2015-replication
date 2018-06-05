# Copyright (C) 2016  The Regents of the University of Michigan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see [http://www.gnu.org/licenses/].


# note: this file needs to be modified to work on remote data files; currently implemented only for reading/writing locally

'''
Takes gzipped Coursera clickstream/log files as input and returns a set of csvs into current working directory.

Each weekly csv is a list of users (rows), with columns corresponding to the features for that week. 
Features come from Mi and Yeung (2015).
'''

import gzip, argparse, json, re, math, datetime, os, bisect
import pandas as pd

MILLISECONDS_IN_SECOND = 1000


def fetch_start_end_date(course_name, run, date_csv = "coursera_course_dates.csv"):
    """
    Fetch course start end end date (so user doesn't have to specify them directly).
    :param course_name: Short name of course.
    :param run: run number
    :param date_csv: Path to csv of course start/end dates.
    :return: tuple of datetime objects (course_start, course_end)
    """
    full_course_name = '{0}-{1}'.format(course_name, run)
    date_df = pd.read_csv(date_csv, usecols=[0, 2, 3]).set_index('course')
    course_start = datetime.datetime.strptime(date_df.loc[full_course_name].start_date, '%m/%d/%y')
    course_end = datetime.datetime.strptime(date_df.loc[full_course_name].end_date, '%m/%d/%y')
    return (course_start, course_end)


def course_len(course_start, course_end):
    '''
    Return the duration of a course, in number of whole weeks.
    Note: Final week may be less than 7 days, depending on course start and end dates.
    :param course_start: datetime object for first day of course (generated from user input)
    :param course_end: datetime object for last day of course (generated from user input)
    :return: integer of course duration in number of weeks (rounded up if necessary)
    '''
    course_start, course_end = course_start, course_end
    n_days = (course_end - course_start).days
    n_weeks = math.ceil(n_days / 7)
    return n_weeks


def timestamp_week(timestamp, course_start, course_end):
    '''
    Get (zero-indexed) week number for a given timestamp.

    :param timestamp: UTC timestamp, in seconds.
    :param course_start: datetime object for first day of course (generated from user input)
    :param course_end: datetime object for last day of course (generated from user input)
    :return: integer week number of timestamp. If week not in range of course dates provided, return None.
    '''

    timestamp = datetime.datetime.fromtimestamp(timestamp / MILLISECONDS_IN_SECOND)
    n_weeks = course_len(course_start, course_end)
    week_starts = [course_start + datetime.timedelta(days=x)
                   for x in range(0, n_weeks * 7, 7)]
    week_number = bisect.bisect_left(week_starts, timestamp) - 1
    if week_number >= 0 and week_number <= n_weeks:
        return week_number
    else: # event is not within official course dates
        return None


def extract_users_dropouts(coursera_clickstream_file, course_start, course_end):
    '''
    Assemble list of all users, and dictionary of their dropout weeks.

    :param coursera_clickstream_file: gzipped Coursera clickstream file; see ./sampledata for example
    :param course_start: datetime object for first day of course (generated from user input)
    :param course_end: datetime object for last day of course (generated from user input)
    :return: tuple of (users, dropout_dict):
        users: Python set of all unique user IDs that registered any activity in clickstream log
        df_dropout: pd.DataFrame of userID, dropout_week for each user (dropout_week = 0 if no valid activity)

    '''
    users = set()
    user_dropout_weeks = {}
    linecount = 0
    with(gzip.open(coursera_clickstream_file, 'r')) as f:
        for line in f:
            try:
                log_entry = json.loads(line.decode("utf-8"))
                user = log_entry.get('username')
                timestamp = log_entry.get('timestamp', 0)
                week = timestamp_week(timestamp, course_start, course_end)
                users.add(user)
            except ValueError as e1:
                print('Warning: invalid log line {0}: {1}'.format(linecount, e1))
            except Exception as e:
                print('Warning: invalid log line {0}: {1}\n{2}'.format(linecount, e, line))
            if user not in user_dropout_weeks.keys(): #no entry for user
                if not week: #current entry is outside valid course dates; initialize entry with dropout_week = 0
                    user_dropout_weeks[user] = 0
                else: #current entry is within valid course dates; initialize entry with dropout_week = week
                    user_dropout_weeks[user] = week
            else: #entry already exists for user; check and update if necessary
                # update entry for user if week is valid and more recent than current entry
                if week and user_dropout_weeks[user] < week:
                    user_dropout_weeks[user] = week
            linecount += 1
    df_dropout = pd.DataFrame.from_dict(user_dropout_weeks, orient='index')
    #rename columns; handled this way because DataFrame.from_dict doesn't support column naming directly
    df_dropout.index.names = ['userID']
    df_dropout.columns = ['dropout_week']
    output = (users, df_dropout)
    return output


def forum_line_proc(line, forumviews, linecount):
    fre = re.compile('/forum/')
    try:
        l = json.loads(line.decode("utf-8"))
        if l['key'] == 'pageview' and fre.search(l['page_url']):
            forumviews.append(l)
    except ValueError as e1:
        print('Warning: invalid log line {0}: {1}'.format(linecount, e1))
    except Exception as e:
        print('Warning: invalid log line {0}: {1}\n{2}'.format(linecount, e, line))
    return forumviews


def quizattempt_line_proc(line, course_start, course_end, quiz_output, linecount):
    qre = re.compile('/quiz/attempt')  # in 'url';avoids counting /quiz/feedback
    try:
        j = json.loads(line.decode("utf-8"))
        user = j.get('username')
        timestamp = j.get('timestamp')
        week = timestamp_week(timestamp, course_start, course_end)
        if week:
            # check if access_type is one of an assessment type, and if it is
            # then append entry of that type to quiz_output[user][week]
            if j.get('key') == 'pageview' and qre.search(j.get('page_url')):
                quiz_output[user][week].append('quizzes_quiz_attempt')
    except ValueError as e1:
        print('Warning: invalid log line {0}: {1}'.format(linecount, e1))
    except Exception as e:
        print('Warning: invalid log line {0}: {1}\n{2}'.format(linecount, e, line))
    return quiz_output


def extract_forum_views_and_quiz_attempts(coursera_clickstream_file, users, course_start, course_end):
    """
    Extract forum views, and quiz views in a single pass.
    :return: 
    """
    # initialize all data structures
    n_weeks = course_len(course_start, course_end)
    forum_output = {user: {n: [] for n in range(n_weeks + 1)} for user in users} # nested dict in format {user: {week: [url1, url2, url3...]}}
    forumviews = []
    linecount = 1
    # compile regex for assessment types
    quiz_output = {user: {n: [] for n in range(n_weeks + 1)} for user in users}  # nested dict in format {user: {week: [accessType, accessType...]}}
    # process each clickstream line, extracting any forum views, active days, or quiz views
    with gzip.open(coursera_clickstream_file, 'r') as f:
        for line in f:
            forumviews = forum_line_proc(line, forumviews, linecount)
            quiz_output = quizattempt_line_proc(line, course_start, course_end, quiz_output, linecount)
            linecount += 1
    # post-process data from each forum: add each forumview URL accessed to (user, week) entry in forum_output
    for p in forumviews:
        user = p.get('username')
        url = p.get('page_url')
        timestamp = p.get('timestamp')
        week = timestamp_week(timestamp, course_start, course_end)
        if week:  # if week falls within active dates of course, add to user entry
            forum_output[user][week].append(url)
    forum_output_list = [(k, week, len(views)) for k, v in forum_output.items() for week, views in v.items()]
    df_forum = pd.DataFrame(data=forum_output_list, columns=['userID', 'week', 'forum_views']).set_index('userID')
    # Quiz
    quiz_view_list = [(user, week, access.count('quizzes_quiz_attempt')) for user, user_data in quiz_output.items() for week, access in user_data.items()]
    df_quiz = pd.DataFrame(quiz_view_list, columns=['userID', 'week', 'quiz_attempts']).set_index('userID')
    return (df_forum, df_quiz)


def initialize_user_week_df(users, course_start, course_end):
    """
    Creates a blank DataFrame with index values for every user/week combination. This can be used to ensure all feature sets include all users and weeks.
    :param users: list of session_user_ids (as string)
    :param course_start: datetime.datetime object representing course start date
    :param course_end: datetime.datetime object representing course end date
    :return: pd.DataFrame with MultiIndex of (session_user_id, week) and no other column values.
    """
    n_weeks = course_len(course_start, course_end)
    weeks = [x for x in range(n_weeks + 1)]
    # create dataframe with entry for every user and week
    user_week_df = pd.DataFrame.from_dict({u: weeks for u in users}, orient="index") \
        .reset_index() \
        .melt(id_vars="index") \
        .drop(columns="variable") \
        .rename(columns={"index": "userID", "value": "week"}) \
        .set_index(["userID", "week"])
    return user_week_df


def extract_weekly_activity_counts_from_csv(csvfile, users, course_start, course_end, count_col_name, time_col_name = "time"):
    """
    Generate dataframe of counts of activity by user and week.
    :param csvfile: path to file of [session_user_id, timestamp] csv file of events.
    :param users: list of session_user_ids (as string)
    :param course_start: datetime.datetime object representing course start date
    :param course_end: datetime.datetime object representing course end date
    :return: pd.DataFrame with columns: session_user_id (obj),  week (int), count_col_name (float) where session_user_id and week are MultiIndex
    """
    temp = initialize_user_week_df(users, course_start, course_end)
    df_in = pd.read_csv(csvfile)
    df_in["week"] = df_in[time_col_name].apply(lambda x: timestamp_week(x * MILLISECONDS_IN_SECOND, course_start, course_end))
    df_in = df_in.groupby(["session_user_id", "week"])\
        .size()\
        .reset_index()\
        .rename(columns = {0:count_col_name, "session_user_id": "userID"})\
        .set_index(["userID", "week"])
    df_out = temp.merge(df_in, how="left", left_index = True, right_index = True).fillna(0)
    return df_out


def generate_threads_started(users, forumfile, commentfile, course_start, course_end):
    """
    Generate counts of threads started, by user and week.
    :param df: pd.DataFrame of forum post data.
    :return: pd.DataFrame of 'session_user_id', 'week', and threads_started.
    """
    temp = initialize_user_week_df(users, course_start, course_end)
    ## todo: read forum and comment data into forum_df
    forum_only_df = pd.read_csv(forumfile)[["thread_id", "post_time", "session_user_id"]]
    comment_only_df = pd.read_csv(commentfile)[["thread_id", "post_time", "session_user_id"]]
    forum_df = pd.concat([forum_only_df, comment_only_df])
    forum_df['week'] = (forum_df['post_time'] * 1000).apply(timestamp_week, args=(course_start, course_end))
    # generate column with thread_order
    forum_df.sort_values(by=['thread_id', 'post_time'], inplace=True)
    forum_df['thread_order'] = forum_df.groupby(['thread_id'])['thread_id'].rank(method='first')
    # get counts of thread started
    df_starts = forum_df[forum_df.thread_order == 1]\
        .groupby(['session_user_id', 'week'])\
        .size()\
        .rename('threads_started')\
        .reset_index().rename(columns = {"session_user_id" : "userID"})
    df_starts.set_index(["userID", "week"], inplace = True)
    ## merge onto temp and fillna(0) to ensure every user/week combination is represented
    df_out = temp.merge(df_starts, how="left", left_index=True, right_index=True).fillna(0)
    return df_out


def generate_appended_xing_csv(df_in, week):
    """
    Create appended feature set for week from df_in.
    :param df_in: Full pandas.DataFrame of userID, week, and additional features.
    :param week: Week to create appended feature set for (starting at zero, inclusive)
    :return: pandas.DataFrame of appended ('wide') features for weeks in interval [0, week], plus dropout column.
    """
    #initialize output data using week 0; additional weeks will be merged on userID
    for i in range(0, week+1): # append data from weeks 0-current week
        df_to_append = df_in[df_in.week == i].drop('week', axis=1)
        df_to_append = df_to_append\
            .rename(columns = lambda x: 'week_{0}_{1}'.format(str(i), str(x)))\
            .reset_index()
        if i ==0: #nothing to merge on yet; initialize df_app using week 0 features
            df_app = df_to_append.set_index('userID')
        else: #append features by merging to current feature set
            df_app = df_app.reset_index()\
            .merge(df_to_append)\
            .set_index('userID')
    return df_app


def generate_weekly_csv(df_in, n_weeks, out_dir = "/output"):
    """
    Create a series of csv files containing all entries for each week in df_in
    :param df_in: pandas.DataFrame of weekly features to write output for
    :param dropout_weeks: pd.DataFrame of userID, dropout_week for each user
    :return: Nothing returned; writes csv files to /xing_extractor_output
    """
    if not os.path.exists(out_dir):
        print("creating directory {}".format(out_dir))
        os.makedirs(out_dir)
    ## create appended dataframe for features from first n_weeks
    wk_appended_df = generate_appended_xing_csv(df_in, n_weeks)
    wk_app_destfile = os.path.join(out_dir, "week_%s_feats.csv" % n_weeks)
    wk_appended_df.to_csv(wk_app_destfile)


def extract_features(coursera_clickstream_file, forumfile, commentfile, lecturedlfile, lectureviewfile, users, course_start, course_end):
    """
    Extract full set of features from clickstream, forum, and comment file.
    :param coursera_clickstream_file: gzipped Coursera clickstream file
    :param forumfile: csv generated by feature_extraction/sql_utils.py for example
    :param commentfile: csv generated by eature_extraction/sql_utils.py
    :param lecturedlfile: csv generated by lecture video download query. See feature_extraction/sql_utils.py
    :param lectureviewfile:
    :param users: list of all user IDs, from extract_users_dropouts(), to count forum views for
    :param course_start: datetime object for first day of course (generated from user input)
    :param course_end: datetime object for last day of course (generated from user input)
    :return: pandas.DataFrame of features by user id and week
    """
    print("Extracting forum views and quiz attempts...")
    forum_views, quiz_attempts = extract_forum_views_and_quiz_attempts(coursera_clickstream_file, users, course_start, course_end)
    print("Complete.\nExtracting forum post and comment counts...")
    forumposts = extract_weekly_activity_counts_from_csv(forumfile, users, course_start, course_end, count_col_name="forum_posts", time_col_name="post_time")
    forumcomments = extract_weekly_activity_counts_from_csv(commentfile, users, course_start, course_end,
                                                         count_col_name="forum_comments", time_col_name="post_time")
    print("Complete. \nExtracting thread starts...")
    thread_starts = generate_threads_started(users, forumfile, commentfile, course_start, course_end)
    print("Complete. \nExtracting lecture downloads/views...")
    lecture_downloads = extract_weekly_activity_counts_from_csv(lecturedlfile, users, course_start, course_end, count_col_name ="lecture_downloads")
    lecture_views = extract_weekly_activity_counts_from_csv(lectureviewfile, users, course_start, course_end,
                                                            count_col_name="lecture_views")
    # merge into single data frame
    features_df = forum_views.reset_index()\
        .merge(quiz_attempts.reset_index())\
        .merge(forumposts.reset_index())\
        .merge(forumcomments.reset_index())\
        .merge(lecture_downloads.reset_index()) \
        .merge(lecture_views.reset_index()) \
        .merge(thread_starts.reset_index()) \
        .set_index('userID')
    return features_df


def main(course_name, run_number, n_weeks, extract_dropout_weeks = False):
    session_dir = "/input/{0}/{1}/".format(course_name, run_number)
    clickstream = [x for x in os.listdir(session_dir) if x.endswith("clickstream_export.gz")][0]
    coursera_clickstream_file = session_dir + clickstream
    forumfile = session_dir + "forum_posts.csv"
    commentfile = session_dir + "forum_comments.csv"
    lecturedlfile = session_dir + "video_downloads.csv"
    lectureviewfile = session_dir + "video_views.csv"
    OUTPUT_DIRECTORY = '/output'
    course_start, course_end = fetch_start_end_date(course_name, run_number, session_dir + 'coursera_course_dates.csv')
    # build features
    print("Extracting users...")
    users, dropout_weeks = extract_users_dropouts(coursera_clickstream_file, course_start, course_end)
    print("Complete. Extracting features...")
    feats_df = extract_features(coursera_clickstream_file, forumfile, commentfile, lecturedlfile, lectureviewfile, users, course_start, course_end)
    # write features to output directory
    generate_weekly_csv(feats_df, n_weeks, out_dir=OUTPUT_DIRECTORY)
    # if desired, write dropout weeks to output directory (extraction of labels is NOT needed for MORF jobs)
    if extract_dropout_weeks:
        dropout_file_path = "%s/user_dropout_weeks.csv" % (OUTPUT_DIRECTORY)
        dropout_weeks.to_csv(dropout_file_path)
    print("Output written to {}".format(OUTPUT_DIRECTORY))


if __name__ == '__main__':
    # build parser
    parser = argparse.ArgumentParser(description='Create features from Coursera clickstream file.')
    parser.add_argument('-n', '--course_name',
                        metavar="course short name [must match name in coursera_course_dates.csv; ex. 'introfinance'",
                        type=str,
                        required=True)
    parser.add_argument('-r', '--run_number', metavar="3-digit run number", type=str, required=True)
    #TODO: add argument for n_weeks here too; this is desired number of weeks to use input data from
    # collect input from parser and assign variables
    args = parser.parse_args()
    main(course_name=args.course_name, run_number=args.run_number)
