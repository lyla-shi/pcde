"""
 Author: Li Shi
 Date: 06/20/2020
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
SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 768
BULK_THRESH = 100


# ------------- scan_dist: horizontal / vertical dist covered by connecting fixn points ---------
# only x or y-coordinates reqd
def calc_scan_dist_hv(fixn_pts_x_y, normalizer=1):
    n = len(fixn_pts_x_y)

    p = fixn_pts_x_y

    lv = [abs(p[i] - p[i - 1]) for i in range(1, n)]
    length = float(sum(lv)) / normalizer

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

    sql_select_scanpath_id = (
        " SELECT DISTINCT a.scanpath_id "
        " FROM sandbox.rdtm_metrics_user_doc_viewnum a "
    )

    sql_select_fixns = (
        " select a.x, a.y "
        " from sandbox.rdtm_fixations a "
        " where a.scanpath_id = %s "
    )

    sql_upd_dur = (
        " UPDATE sandbox.rdtm_metrics_user_doc_viewnum a "
        " SET a.scan_dist_h = %s "
        ", a.scan_dist_v = %s " 
        ", a.scan_hv_ratio = %s"
        " where a.scanpath_id = %s "
    )

    #cursor
    cur_select_scanpath_id = conn.cursor()
    cur_select_fixns = conn.cursor()
    cur_upd_dur = conn.cursor()
    print('Start processing...')

    try:
        bulk_insert = []
        cur_select_scanpath_id.execute(sql_select_scanpath_id)
        i_scanpath = 0

        for scanpath_id in cur_select_scanpath_id:
            id = scanpath_id['scanpath_id']
            i_scanpath += 1

            cur_select_fixns.execute(sql_select_fixns, id)
            list_fixn_x = []
            list_fixn_y = []
            for fixn in cur_select_fixns:
                list_fixn_x.append(fixn['x'])
                list_fixn_y.append(fixn['y'])

            scan_dist_h = calc_scan_dist_hv(list_fixn_x, normalizer=SCREEN_WIDTH)
            scan_dist_v = calc_scan_dist_hv(list_fixn_y, normalizer=SCREEN_HEIGHT)

            scan_hv_ratio = scan_dist_h

            if scan_dist_v > 0:
                scan_hv_ratio = (scan_dist_h / scan_dist_v)
                """
                thus when vertical distance is zero, 
                we assume vertical distance is 1 pixel
                """

            bulk_insert.append([
                float(round(scan_dist_h, 10)),
                float(round(scan_dist_v, 10)),
                float(round(scan_hv_ratio, 10)),
                id
            ])

            if i_scanpath % BULK_THRESH == 0:
                cur_upd_dur.executemany(sql_upd_dur, bulk_insert)
                print("No. of docid_userids inserted = %d" % (i_scanpath))
                bulk_insert = []


        conn.commit()
        print('Done. Finished')



    # ----------- actual code ENDS here -----------------
    finally:
        cur_select_fixns.close()
        conn.close()
        print("end")

        if socket.getfqdn() != 'soi-volt.ischool.utexas.edu':
            tunnel.stop()
            tunnel.close()


if __name__ == '__main__':
    main()
    exit(0)