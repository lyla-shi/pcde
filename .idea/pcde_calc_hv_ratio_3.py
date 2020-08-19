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
BULK_THRESH = 100


# ------------- scan_dist: horizontal / vertical dist covered by connecting fixn points ---------
# only x or y-coordinates reqd (e.g. [116, 232, 104,...]
# SUM(abs(x2 - x1))/normalizer
def calc_scan_dist_hv(list_of_points, normalizer=1):
    n = len(list_of_points)
    p = list_of_points
    length_list = [abs(p[i] - p[i-1]) for i in range(1, n)]
    length = float(sum(length_list)) / normalizer
    return length



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

    sql_select_docuserid = (
        "SELECT DISTINCT DOCID_USERID "
        "FROM sandbox.pcde_fixations a "
        " where 1=1 "
        # " and a.DOCID_USERID = 'NYT19990107.0169_16' "
    )

    sql_select_fixns = (
        " SELECT * "
        " FROM sandbox.pcde_fixations a "
        " WHERE a.DOCID_USERID = %s "
        " AND a.IS_FIXN_CROSSHAIR = 0 "
        " ORDER BY a.`Start` "
    )

    sql_upd_metrics = (
        " UPDATE sandbox.pcde_metrics_user_doc a "
        " SET a.scan_hv_ratio_2 = %s "
        " WHERE a.DOCID_USERID = %s "
    )

    #cursor
    cur_select_docuserid = conn.cursor()
    cur_select_fixns = conn.cursor()
    cur_upd_metrics = conn.cursor()

    print('Start processing...')

    try:
        cur_select_docuserid.execute(sql_select_docuserid)

        bulk_insert = []
        i_docid_userid = 0

        # ---- id loop----
        for row_id in cur_select_docuserid:
            i_docid_userid += 1
            # print("row_id['DOCID_USERID']:", row_id['DOCID_USERID'])
            cur_select_fixns.execute(sql_select_fixns, row_id['DOCID_USERID'])

            fixn_x = []
            fixn_y = []
            # ---- fixations loop with each id----
            for row_fixn in cur_select_fixns:
                fixn_x.append(row_fixn['LocationX'])
                fixn_y.append(row_fixn['LocationY'])
            # -- end fixations ---



            # --- check num of fixations ---
            if len(fixn_x) < 2:
                scan_hv_ratio = 0
                # cur_upd_metrics.execute(sql_upd_metrics, (
                #     0 , row_id['DOCID_USERID']
                # ))
            else:
                # normalizing for stimulus dimensions
                scan_dist_h = calc_scan_dist_hv(fixn_x, normalizer=SCREEN_WIDTH)  # sum hi / w
                scan_dist_v = calc_scan_dist_hv(fixn_y, normalizer=SCREEN_HEIGHT) # sum vi / h
                # (sum hi / W) / (sum vi / H)  OR   (sum hi* H) / (sum vi * W )
                scan_hv_ratio = (scan_dist_h / scan_dist_v)
                # print("Calculated HV ratio:", float(round(scan_hv_ratio, 10)))

            # -- end if else

            # --- preparing bulk insert list ----

            bulk_insert.append([
                float(round(scan_hv_ratio, 10)),
                row_id['DOCID_USERID']
            ])

            if i_docid_userid % BULK_THRESH == 0:
                cur_upd_metrics.executemany(sql_upd_metrics, bulk_insert)
                print("No. of docid_userids inserted = %d" % (i_docid_userid))
                bulk_insert = []

            # cur_upd_metrics.execute(sql_upd_metrics, (
            #     float(round(scan_hv_ratio, 10)), row_id['DOCID_USERID']
            # ))

        # -- end ids loop ---

        # --- bulk insert remaining rows ----
        cur_upd_metrics.executemany(sql_upd_metrics, bulk_insert)
        print("No. of docid_userids inserted = %d" % (i_docid_userid))

        conn.commit()
        print('Done. Finished')

    # ----------- actual code ENDS here -----------------
    finally:
        cur_select_fixns.close()
        conn.close()

        if socket.getfqdn() != 'soi-volt.ischool.utexas.edu':
            tunnel.stop()
            tunnel.close()


if __name__ == '__main__':
    main()
    exit(0)