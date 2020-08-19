import pandas as pd
import numpy as np
#from math import sqrt, atan2
import math
from scipy.stats import hmean
from sshtunnel import SSHTunnelForwarder
import pymysql
import sys, os, socket
from scipy.spatial import ConvexHull

# ssh_host = sys.argv[1]
# ssh_port = int(sys.argv[2])
# ssh_user = sys.argv[3]
# ssh_pass = sys.argv[4]
#
# db_host = sys.argv[5]
# db_port = int(sys.argv[6])
# db_user = sys.argv[7]
# db_pass = sys.argv[8]
# db_name = sys.argv[9]
#
# def main():
#
#     db_port_actual = db_port
#
#     if socket.getfqdn() != 'soi-volt.ischool.utexas.edu':
#
#         tunnel = SSHTunnelForwarder(
#             (ssh_host, ssh_port),
#             ssh_username=ssh_user,
#             ssh_password=ssh_pass,
#             remote_bind_address=(db_host, db_port)
#         )
#
#         tunnel.start()
#         db_port_actual = tunnel.local_bind_port
#         print("Not on volt. SSH tunnel started.")
#
#     print("Connecting to database at port: %d" % db_port_actual)
#
#     conn = pymysql.connect(
#         host=db_host,
#         port=db_port_actual,
#         user=db_user,
#         passwd=db_pass,
#         db=db_name,
#         charset='utf8mb4',
#         cursorclass=pymysql.cursors.DictCursor
#     )
#
#
#     sql_select_fixns = (
#         " select a.fixn_dur "
#         " from sandbox.rdtm_fixations_20200618 a "
#         " where a.userid = %s "
#         " and a.docid = %s "
#         " and a.view_num = %s "
#     )
#
#     sql_select_id = """
#         select a.scanpath_id
#         from sandbox.rdtm_metrics_user_doc_viewnum a
#         where a.userid = %s
#         and a.docid = %s
#         and a.view_num = %s
#     """
#
#     sql_upd_dur1 = (
#         " UPDATE sandbox.rdtm_metrics_user_doc_viewnum a "
#         " SET a.fixn_dur_sum = %s "
#         " where a.scanpath_id = %s "
#     )
#
#     cur_select_fixns = conn.cursor()
#     cur_upd_dur1 = conn.cursor()
#     cur_select_id = conn.cursor()
#
#     print('Start processing...')
#
#     try:
#         cur_select_fixns.execute(sql_select_fixns, (137,1,1))
#
#         list_fixn_dur = []
#         for fixn in cur_select_fixns:
#             list_fixn_dur.append(fixn['fixn_dur'])
#             print(fixn['fixn_dur'])
#
#         fixn_dur_sum = np.nan_to_num(sum(list_fixn_dur))
#         print("fixn_dur_sum", float(round(fixn_dur_sum, 10)))
#
#         cur_select_id.execute(sql_select_id, (137,1,1))
#         for ids in cur_select_id:
#             wantedid = ids['scanpath_id']
#             print ("id", wantedid)
#
#         cur_upd_dur1.execute(sql_upd_dur1, (float(round(12138, 10)), wantedid))
#
#         conn.commit()
#         print('Done. Finished')
#
# # ----------- actual code ENDS here -----------------
#     finally:
#         cur_select_fixns.close()
#         conn.close()
#         print("end")
#
#         if socket.getfqdn() != 'soi-volt.ischool.utexas.edu':
#             tunnel.stop()
#             tunnel.close()
#
#
# if __name__ == '__main__':
#     main()
#     exit(0)

list
l = np.sort(list_of_points)
min_val = np.min(l)
max_val = np.max(l)
up_quartile, low_quartile = np.percentile(l, [75, 25], interpolation='lower')
iqr = up_quartile - low_quartile

low_whisker = max(min_val, low_quartile - 1.5 * iqr)
up_whisker = min(max_val, up_quartile + 1.5 * iqr)
return low_whisker, up_whisker