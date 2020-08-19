"""
 Author:Li Shi
 Date: 06/30/2020
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
        " select a.scan_dist_h, a.scan_dist_v, a.scan_hv_ratio, a.scanpath_id "
        " from sandbox.rdtm_metrics_user_doc_viewnum a "
        " where a.userid = %s "
    )

    sql_upd_dur_p = (
        " UPDATE sandbox.rdtm_metrics_user_doc_viewnum a "
        " SET a.scan_dist_h_p = %s "
        ", a.scan_dist_v_p = %s "
        ", a.scan_hv_ratio_p = %s "
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

            list_dist_h = []
            list_dist_v = []
            list_hv = []
            list_scanid = []
            for fixn in cur_select_fixn_dur:
                list_dist_h.append(fixn['scan_dist_h'])
                list_dist_v.append(fixn['scan_dist_v'])
                list_hv.append(fixn['scan_hv_ratio'])
                list_scanid.append(fixn['scanpath_id'])

            h_lower_whisker, h_upper_whisker = calc_whisker(list_dist_h)
            v_lower_whisker, v_upper_whisker = calc_whisker(list_dist_v)
            hv_lower_whisker, hv_upper_whisker = calc_whisker(list_hv)
            list_fixn_dur_p = []

            for i in range(0, len(list_dist_h)):
                if list_dist_h[i] < h_lower_whisker:
                    normalize_h = 0
                elif list_dist_h[i] > h_upper_whisker:
                    normalize_h = 1
                else:
                    normalize_h = (list_dist_h[i] - h_lower_whisker)/(h_upper_whisker - h_lower_whisker)

                if list_dist_v[i] < v_lower_whisker:
                    normalize_v = 0
                elif list_dist_v[i] > v_upper_whisker:
                    normalize_v = 1
                else:
                    normalize_v = (list_dist_v[i] - v_lower_whisker)/(v_upper_whisker - v_lower_whisker)

                if list_hv[i] < hv_lower_whisker:
                    normalize_hv = 0
                elif list_hv[i] > hv_upper_whisker:
                    normalize_hv = 1
                else:
                    normalize_hv = (list_hv[i] - hv_lower_whisker)/(hv_upper_whisker - hv_lower_whisker)

                #append list
                # print(list_scanid[i])
                # print("whiskers", v_upper_whisker, v_lower_whisker)
                # print ("dist_v", normalize_v)
                list_fixn_dur_p.append([float(round(normalize_h, 10)),float(round(normalize_v, 10)), float(round(normalize_hv, 10)), list_scanid[i]])

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