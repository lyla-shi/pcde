"""
 Author: Li Shi
 Date: 2020-06-11
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



# ------------- scan_dist: horizontal / vertical dist covered by connecting fixn points ---------
# only x or y-coordinates read (e.g. [116, 232, 104,...]
# SUM(abs(x2 - x1))/normalizer
def calc_scan_dist_hv(list_of_points, normalizer=1):
    n = len(list_of_points)
    p = list_of_points
    length_list = [abs(p[i] - p[i-1]) for i in range(1, n)]
    length = float(sum(length_list)) / normalizer
    return length



# -------------------- main() -------------------------------
def main():
    tunnel = SSHTunnelForwarder(
        (ssh_host, ssh_port),
        ssh_username=ssh_user,
        ssh_password=ssh_pass,
        remote_bind_address=(db_host, db_port)
    )

    tunnel.start()

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

    sql_select_fixns = (
        " SELECT * "
        " FROM sandbox.pcde_fixations a "
        # " WHERE a.DOCID_USERID = 'NYT19990420.0077_21' "
        " WHERE a.DOCID_USERID = 'APW19990818.0004_1' "
        " AND a.IS_FIXN_CROSSHAIR = 0 "
        " ORDER BY a.`Start` "

    )

    sql_upd_metrics = (
        " UPDATE sandbox.pcde_metrics_user_doc a "
        " SET a.scan_hv_ratio_2 = %s "
        " WHERE a.DOCID_USERID = 'APW19990818.0004_1' "
    )

    #cursor

    cur_select_fixns = conn.cursor()
    cur_upd_metrics = conn.cursor()

    print('Start processing...')

    try:
        cur_select_fixns.execute(sql_select_fixns)


        fixn_x = []
        fixn_y = []
        # ---- fixations loop ----
        for row_fixn in cur_select_fixns:
            print("Now at fixation point:", row_fixn['LocationX'], row_fixn['LocationY'])

            fixn_x.append(row_fixn['LocationX'])
            fixn_y.append(row_fixn['LocationY'])
        # -- end fixations ---

        # normalizing for stimulus dimensions
        scan_dist_h = calc_scan_dist_hv(fixn_x, normalizer=SCREEN_WIDTH)  # sum hi / H w
        scan_dist_v = calc_scan_dist_hv(fixn_y, normalizer=SCREEN_HEIGHT)
        # sum hi / H w
        # sum vi / V h
        # (sum hi / H) / (sum vi / V)  OR   (sum hi* V) / (sum vi * H)

        scan_hv_ratio = (scan_dist_h / scan_dist_v)
        print("Calculated HV ratio:", scan_hv_ratio, 10)

        cur_upd_metrics.execute(sql_upd_metrics, (
            float(round(scan_hv_ratio, 10))
        ))






        # ----------- actual code ENDS here -----------------
        conn.commit()
        print('Done. Finished')

    finally:
        cur_select_fixns.close()

        conn.close()
        tunnel.stop()
        tunnel.close()


if __name__ == '__main__':
    main()
    exit(0)