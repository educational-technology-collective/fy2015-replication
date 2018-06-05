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

import os
import subprocess


DATABASE_NAME = "course"

def execute_mysql_query_into_csv(query, file, database_name=DATABASE_NAME):
    """
    Execute a mysql query into a file.
    :param query: valid mySQL query as string.
    :param file: csv filename to write to.
    :return: none
    """
    mysql_to_csv_cmd = """ | tr '\t' ',' """  # string to properly format result of mysql query
    command = '''mysql -u root -proot {} -e"{}"'''.format(database_name, query)
    command += """{} > {}""".format(mysql_to_csv_cmd, file)
    subprocess.call(command, shell=True)
    return


def load_dump(dump_file, dbname = DATABASE_NAME):
    print("[INFO] loading dump from {}".format(dump_file))
    command = '''mysql -u root -proot {} < {}'''.format(dbname, dump_file)
    res = subprocess.call(command, shell=True)
    print("[INFO] result: {}".format(res))
    return

def load_data(course, session, dbname = DATABASE_NAME):
    """
    Loads data into mySQL database from database dump files.
    :param course: shortname of course.
    :param session: 3-digit session id (string).
    :return:
    """
    password = 'root'
    user = 'root'
    mysql_binary_location = '/usr/bin/mysql'
    mysql_admin_binary_location = '/usr/bin/mysqladmin'
    hash_mapping_sql_dump = \
    [x for x in os.listdir('/input/{}/{}'.format(course, session)) if 'hash_mapping' in x and session in x][0]
    forum_sql_dump = \
    [x for x in os.listdir('/input/{}/{}'.format(course, session)) if 'anonymized_forum' in x and session in x][0]
    anon_general_sql_dump = \
    [x for x in os.listdir('/input/{}/{}'.format(course, session)) if 'anonymized_general' in x and session in x][0]
    # start mysql server
    subprocess.call('service mysql start', shell=True)
    # create a database
    print("[INFO] creating database")
    res = subprocess.call('''mysql -u root -proot -e "CREATE DATABASE {}"'''.format(dbname), shell=True)
    print("RES: {}".format(res))
    # load all data dumps needed
    load_dump("/input/{}/{}/{}".format(course, session, forum_sql_dump))
    load_dump("/input/{}/{}/{}".format(course, session, hash_mapping_sql_dump))
    load_dump("/input/{}/{}/{}".format(course, session, anon_general_sql_dump))
    return


def extract_coursera_sql_data(course, session):
    '''
    Initializes the MySQL database. This assumes that MySQL is correctly setup in the docker container.
    :return:
    '''
    load_data(course, session)
    course_session_dir = os.path.join("input", course, session)
    # execute forum comment query and send to csv
    query = """SELECT thread_id , post_time , b.session_user_id FROM forum_comments as a LEFT JOIN hash_mapping as b ON a.user_id = b.user_id WHERE a.is_spam != 1 ORDER BY post_time;"""
    execute_mysql_query_into_csv(query, os.path.join(course_session_dir, "forum_comments.csv"))
    # execute forum post query and send to csv
    query = """SELECT id , thread_id , post_time , a.user_id , public_user_id , session_user_id , eventing_user_id FROM forum_posts as a LEFT JOIN hash_mapping as b ON a.user_id = b.user_id WHERE is_spam != 1 ORDER BY post_time;"""
    execute_mysql_query_into_csv(query, os.path.join(course_session_dir, "forum_posts.csv"))
    # execute video download query and sent to csv
    query = """select session_user_id, submission_time AS time from lecture_submission_metadata WHERE action = 'download';"""
    execute_mysql_query_into_csv(query, os.path.join(course_session_dir, "video_downloads.csv"))
    # execute lecture view query and send to csv
    query = """select session_user_id, submission_time AS time from lecture_submission_metadata WHERE action = 'view';"""
    execute_mysql_query_into_csv(query, os.path.join(course_session_dir, "video_views.csv"))
    return None
