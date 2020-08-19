"""
 Author:Li Shi
 Date: 06/29/2020
"""
import pandas as pd
import numpy as np
#from math import sqrt, atan2
import math
from scipy.stats import hmean
from sshtunnel import SSHTunnelForwarder
import pymysql
import sys, os, socket
from scipy.spatial import ConvexHull

# -------------------- CONNECTION PARAMS -------------------------------
ssh_host = sys.argv[1]
ssh_port = int(sys.argv[2])
ssh_user = sys.argv[3]
ssh_pass = sys.argv[4]

db_host = sys.argv[5]
db_port = int(sys.argv[6])
db_user = sys.argv[7]
db_pass = sys.argv[8]
db_name = sys.argv[9]

# -------------------- processing params ---------------------------
# SCREEN_WIDTH = 1680
# SCREEN_HEIGHT = 1050
BULK_THRESH = 100

# ------------- calculate whisker ---------
# lower whisker = max(minimum value, lower quartile − 1.5 * interquartile range);
# upper whisker = min(maximum value, upper quartile + 1.5 * interquartile range);
def calc_whisker(list_of_points):
    l = np.sort(list_of_points)
    min_val = np.min(l)
    max_val = np.max(l)
    up_quartile, low_quartile = np.percentile(l, [75, 25], interpolation='lower')
    iqr = up_quartile - low_quartile

    low_whisker = max(min_val, low_quartile - 1.5 * iqr)
    up_whisker = min(max_val, up_quartile + 1.5 * iqr)
    return low_whisker, up_whisker


# -------------------- main() -------------------------------
def main():

    db_port_actual = db_port

    if socket.getfqdn() != 'soi-volt.ischool.utexas.edu':

        tunnel = SSHTunnelForwarder(
            (ssh_host, ssh_port),
            ssh_username=ssh_user,
            ssh_password=ssh_pass,
            remote_bind_address=(db_host, db_port)
        )

        tunnel.start()
        db_port_actual = tunnel.local_bind_port
        print("Not on volt. SSH tunnel started.")

    print("Connecting to database at port: %d" % db_port_actual)

    conn = pymysql.connect(
        host=db_host,
        port=db_port_actual,
        user=db_user,
        passwd=db_pass,
        db=db_name,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )


    # ----------- actual code starts from here -----------------

    sql_select_userid = (
        " SELECT DISTINCT a.userid "
        " FROM sandbox.rdtm_metrics_user_doc_viewnum a "
    )

    sql_select_fixn_dur = (
        " select a.fixn_dur_sum, a.scanpath_id "
        " from sandbox.rdtm_metrics_user_doc_viewnum a "
        " where a.userid = %s "
        " order by a.fixn_dur_sum "
    )

    sql_upd_dur_p = (
        " UPDATE sandbox.rdtm_metrics_user_doc_viewnum a "
        " SET a.fixn_dur_sum_p = %s "
        " where a.scanpath_id = %s "
    )

    #cursor
    cur_select_userid = conn.cursor()
    cur_select_fixn_dur = conn.cursor()
    cur_upd_dur_p = conn.cursor()

    print('Start processing...')

    try:
        bulk_insert = []
        cur_select_userid.execute(sql_select_userid)

        for userid in cur_select_userid:

            cur_select_fixn_dur.execute(sql_select_fixn_dur, userid['userid'])

            list_fixn_dur = []
            list_scanid = []
            for fixn in cur_select_fixn_dur:
                list_fixn_dur.append(fixn['fixn_dur_sum'])
                list_scanid.append(fixn['scanpath_id'])

            lower_whisker, upper_whisker = calc_whisker(list_fixn_dur)
            list_fixn_dur_p = []

            for i in range(0, len(list_fixn_dur)):
                if list_fixn_dur[i] < lower_whisker:
                    normalize = 0
                elif list_fixn_dur[i] > upper_whisker:
                    normalize = 1
                else:
                    normalize = (list_fixn_dur[i] - lower_whisker)/(upper_whisker - lower_whisker)
                #append list
                list_fixn_dur_p.append([float(round(normalize, 10)), list_scanid[i]])

            # update for each user
            cur_upd_dur_p.executemany(sql_upd_dur_p, list_fixn_dur_p)
            print("user", userid['userid'], "updated;")

        conn.commit()
        print('Done. Finished')

    # ----------- actual code ENDS here -----------------
    finally:
        cur_select_fixn_dur.close()
        cur_select_userid.close()
        cur_upd_dur_p.close()
        conn.close()
        print("end")

        if socket.getfqdn() != 'soi-volt.ischool.utexas.edu':
            tunnel.stop()
            tunnel.close()


if __name__ == '__main__':
    main()
    exit(0)