"""
 Author: Li Shi
 Date: 2020-06-22
 Author URL: <>
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
SCREEN_WIDTH = 1680
SCREEN_HEIGHT = 1050



# ------------- calculate whisker ---------
# lower whisker = max(minimum value, lower quartile − 1.5 * interquartile range);
# upper whisker = min(maximum value, upper quartile + 1.5 * interquartile range);
def calc_whisker2(list_of_points):
    l = np.sort(list_of_points)
    min_val = np.min(l)
    max_val = np.max(l)
    up_quartile, low_quartile = np.percentile(l, [75, 25], interpolation='lower')
    iqr = up_quartile - low_quartile

    low_whisker = max(min_val, low_quartile - 1.5 * iqr)
    up_whisker = min(max_val, up_quartile + 1.5 * iqr)
    return low_whisker, up_whisker


def calc_whisker(list_of_points):
    n = len(list_of_points)
    min_val = list_of_points[0]
    max_val = list_of_points[n-1]
    low_quartile = list_of_points[math.floor(n/4) -1]
    up_quartile = list_of_points[math.floor(n/4)*3 -1]
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
        port=tunnel.local_bind_port,
        user=db_user,
        passwd=db_pass,
        db=db_name,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )


    # ----------- actual code starts from here -----------------

    sql_select_userid = (
        " SELECT DISTINCT a.userid "
        " FROM sandbox.pcde_metrics_user_doc a "
    )

    sql_select_hv = (
        " select a.scan_hv_ratio_2, a.DOCID_USERID "
        " from sandbox.pcde_metrics_user_doc a "
        " where a.userid = %s "
        " order by a.scan_hv_ratio_2 "
    )

    sql_upd_metrics = (
        " UPDATE sandbox.pcde_metrics_user_doc a "
        #" SET a.scan_hv_ratio_2_personalized = %s "
        " SET a.scan_hv_ratio_2_pers2 = %s "
        " WHERE a.DOCID_USERID = %s "
    )

    #cursor
    cur_select_userid = conn.cursor()
    cur_select_hv = conn.cursor()
    cur_upd_metrics = conn.cursor()

    print('Start processing...')

    try:
        cur_select_userid.execute(sql_select_userid)

        bulk_insert = []
        i_user = 0

        # ---- user id loop----
        for row_id in cur_select_userid:
            i_user += 1
            # get hv_ratio and user_docid for each user
            cur_select_hv.execute(sql_select_hv, row_id['userid'])

            hv = []
            user_docid = []
            personalized_hv = []

            # user loop to save hv_ratio and user_docid
            for row_hv in cur_select_hv:
                hv.append(row_hv['scan_hv_ratio_2'])
                user_docid.append(row_hv['DOCID_USERID'])

            #-----whisker----
            lower_whisker, upper_whisker = calc_whisker(hv)

            #-----calculate for each docid_userid ------
            # normalized with respect to the individual whisker-intervals
            for i in range(0, len(hv)):
                if hv[i] < lower_whisker:
                    normalize = 0
                elif hv[i] > upper_whisker:
                    normalize = 1
                else:
                    normalize = (hv[i] - lower_whisker)/(upper_whisker - lower_whisker)
                #append list
                personalized_hv.append([float(round(normalize, 10)), user_docid[i]])

            # update for each user
            cur_upd_metrics.executemany(sql_upd_metrics, personalized_hv)
            print(i_user, "updated;")
        # -- end ids loop ---

        conn.commit()
        print('Done. Finished')

    # ----------- actual code ENDS here -----------------
    finally:
        cur_select_hv.close()
        conn.close()

        if socket.getfqdn() != 'soi-volt.ischool.utexas.edu':
            tunnel.stop()
            tunnel.close()


if __name__ == '__main__':
    main()
    exit(0)