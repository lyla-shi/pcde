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
# SCREEN_WIDTH = 1680
# SCREEN_HEIGHT = 1050
BULK_THRESH = 100


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
        " select a.fixn_dur "
        " from sandbox.rdtm_fixations a "
        " where a.scanpath_id = %s "
    )

    sql_upd_dur = (
        " UPDATE sandbox.rdtm_metrics_user_doc_viewnum a "
        " SET a.fixn_dur_sum = %s "
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

            list_fixn_dur = []
            for fixn in cur_select_fixns:
                list_fixn_dur.append(fixn['fixn_dur'])
            fixn_dur_sum = np.nan_to_num(sum(list_fixn_dur))
            # print ("fixn_dur_sum",fixn_dur_sum)

            bulk_insert.append([
                float(round(fixn_dur_sum, 10)),
                id
            ])

            if i_scanpath % BULK_THRESH == 0:
                cur_upd_dur.executemany(sql_upd_dur, bulk_insert)
                print("No. of docid_userids inserted = %d" % (i_scanpath))
                bulk_insert = []


        # cur_select_id.execute(sql_select_id)
        # for id in cur_select_id:
        #     print(id['scanpath_id'])
        #     unique_id = str(id['scanpath_id'])

        #
        # cur_select_fixns.execute(sql_select_fixns, (137,1,1))
        #
        # list_fixn_dur = []
        # for fixn in cur_select_fixns:
        #     list_fixn_dur.append(fixn['fixn_dur'])
        #
        # fixn_dur_sum = np.nan_to_num(sum(list_fixn_dur))
        # print ("fixn_dur_sum", float(round(fixn_dur_sum, 10)))
        #
        # cur_upd_dur1.execute(sql_upd_dur1, float(123456))


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