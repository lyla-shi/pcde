"""
 Author: Nilavra Bhattacharya
 Date: 2019-02-21
 Author URL: https://nilavra.in
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
MILLI_FACTOR = 0.001 # to convert from microseconds to milliseconds
SEC_FACTOR = MILLI_FACTOR * 0.001 # # to convert from microseconds to seconds

SCREEN_WIDTH = 1680
SCREEN_HEIGHT = 1050



# ------------- scan_dist: euclidean distance covered by connecting fixn points ---------
def calc_scan_dist_euclid(fixn_pts_x, fixn_pts_y, normalizer_x=1, normalizer_y=1):
    n = len(fixn_pts_x)

    x = fixn_pts_x #[fixn_pts[i][0] for i in range(n)]
    y = fixn_pts_y #[fixn_pts[i][1] for i in range(n)]

    lv = [math.sqrt(
        ((x[i] - x[i - 1]) / normalizer_x) ** 2  +
        ((y[i] - y[i - 1]) / normalizer_y) ** 2
    ) for i in range(1, n)]

    length = sum(lv)
    return length

# ------------- scan_dist: horizontal / vertical dist covered by connecting fixn points ---------
# only x or y-coordinates reqd
def calc_scan_dist_hv(fixn_pts_x_y, normalizer=1):
    n = len(fixn_pts_x_y)

    p = fixn_pts_x_y

    lv = [abs(p[i] - p[i - 1]) for i in range(1, n)]
    length = float(sum(lv)) / normalizer

    return length

# ---------------- convexHullArea2D: area of convex hull of points ---------------
# pts is in the form [ (1,1), (2,1), (5,2), (3,4) ]
def calc_convex_hull_area_centroid(fixn_pts_x, fixn_pts_y):

    pts = []
    for i in range(len(fixn_pts_x)):
        pts.append([fixn_pts_x[i], fixn_pts_y[i]])

    hull = ConvexHull(pts)

    # Get centroid
    cx = np.mean(hull.points[hull.vertices, 0])
    cy = np.mean(hull.points[hull.vertices, 1])

    # in 2D, area = perimeter, volume = area -- https://stackoverflow.com/a/46246955
    return hull.volume, cx, cy


# ------------- calculates r (length) and theta (angle) of overall fixation vector, w.r.t. a central point  ---------
def calc_overall_fixn_vector(
        fixn_pts_x, fixn_pts_y, fixn_dur_list, x_c=0.0, y_c=0.0,
        normalizer_x=1, normalizer_y=1
):

    n = len(fixn_pts_x)

    x = fixn_pts_x #[fixn_pts[i][0] for i in range(n)]
    y = fixn_pts_y #[fixn_pts[i][1] for i in range(n)]
    dur = fixn_dur_list

    lx = [((float(x[i]) - x_c) / normalizer_x) * dur[i] for i in range(0, n)]
    ly = [((float(y[i]) - y_c) / normalizer_y) * dur[i] for i in range(0, n)]

    x_vec = sum(lx)
    y_vec = sum(ly)

    r = math.sqrt((x_vec - x_c) ** 2 + (y_vec - y_c) ** 2)
    # atan2 returns angles in [-pi to pi]. Adding pi changes this to [0 to pi]
    theta_rad = math.atan2(y_vec, x_vec) #+ math.pi
    theta_deg = math.degrees(theta_rad)

    return x_vec, y_vec, r, theta_rad, theta_deg


# -------------------- main() -------------------------------

def main():

    BULK_THRESH = 100

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

    sql_insert = (
        " insert into sandbox.pcde_metrics_user_doc ("
        "     userid "
        "   , docid "
        "   , DOCID_USERID "
        "   , p_rlvnc "
        "   , t_rlvnc "
        "   , task_st "
        "   , task_end "
        "   , task_dur "
        "   , first_fixn_st "
        "   , last_fixn_end "
        "   , task_fixn_dur "
        "   , fixn_n "

        "   , fixn_n_l1 "
        "   , fixn_n_l2 "
        "   , fixn_n_l3 "
        "   , fixn_n_l4 "

        "   , fixn_rate "
        "   , fixn_dur_sum "
        "   , fixn_dur_avg "

        "   , fixn_dur_sd "

        "   , fixn_hull_area "
        "   , fixn_hull_x "
        "   , fixn_hull_y "
        "   , scan_dist_euclid "
        "   , scan_dist_h "
        "   , scan_dist_v "
        "   , scan_hv_ratio "
        "   , scan_speed "
        "   , scan_speed_h "
        "   , scan_speed_v "
        "   , fixn_per_scan_dist "

        "   , avg_sacc_len "

        "   , overall_fixn_vec0_x "
        "   , overall_fixn_vec0_y "
        "   , overall_fixn_vec0_len "
        "   , overall_fixn_vec0_theta_rad "
        "   , overall_fixn_vec0_theta_deg "
        "   , overall_fixn_vec_x "
        "   , overall_fixn_vec_y "
        "   , overall_fixn_vec_len "
        "   , overall_fixn_vec_theta_rad "
        "   , overall_fixn_vec_theta_deg "
        ") values ("
        "  %s, %s, %s, %s, %s, "
        "  %s, %s, %s, %s, %s, "
        "  %s, %s, %s, %s, %s, "
        "  %s, %s, %s, %s, %s, "
        "  %s, %s, %s, %s, %s, "
        "  %s, %s, %s, %s, %s, "
        "  %s, %s, %s, %s, %s, "
        "  %s, %s, %s, %s, %s, "
        "  %s, %s "
        ")"
    )

    sql_select_usr_doc = (
        " select distinct a.Userid userid "
        "     , a.DocId docid "
        "     , a.`Start` "
        "     , a.`End` "
        "     , b.t_rlvnc "
        "     , b.p_rlvnc "
        " from PCDE.TextStimsOne a "
        " , sandbox.pcde_fixations b "
        " where a.Userid = b.userid "
        " AND a.DocId = b.docid "
        " order by a.Userid, a.DocId "
    )


    sql_select_fixns = (
        " select * "
        " from sandbox.pcde_fixations a "
        " where a.userid = %s "
        " and a.docid = %s "
        " and a.IS_FIXN_CROSSHAIR <> 1 " # ignore fixations outside text area
        " order by a.`Start` "
    )


    cur_select_usr_doc = conn.cursor()
    cur_select_fixns = conn.cursor()
    cur_insert = conn.cursor()


    print('Start processing...')

    try:

        bulk_insert = []

        cur_select_usr_doc.execute(sql_select_usr_doc)

        tot_user_doc = int(cur_select_usr_doc.rowcount)
        i_user_doc = 0

        tot_rows_insert = 0

        ################
        # user-doc loop
        ################
        for user_doc in cur_select_usr_doc:

            userid = user_doc['userid']
            docid = user_doc['docid']

            DOCID_USERID = str(docid) + "_" + str(userid)

            p_rlvnc = user_doc['p_rlvnc']
            t_rlvnc = user_doc['t_rlvnc']



            task_st = user_doc['Start'] # in microseconds
            task_end = user_doc['End'] # in microseconds
            task_dur = 1. * (task_end - task_st) * SEC_FACTOR # in seconds

            i_user_doc += 1

            # print('---------- userid: %s \t docid: %s \t [%3d / %3d] ----------'
            #       % (userid, docid, i_user_doc, tot_user_doc))

            cur_select_fixns.execute(sql_select_fixns, (userid, docid))

            fixn_n = int(cur_select_fixns.rowcount)

            fixn_rate = fixn_n / task_dur

            i_fixn = 0

            list_fixn_dur = []
            list_fixn_x = []
            list_fixn_y = []

            ######################
            # fixations loop
            ######################
            for fixn in cur_select_fixns:

                i_fixn += 1

                if i_fixn == 1:
                    first_fixn_st = fixn['Start']

                if i_fixn == fixn_n:
                    last_fixn_end = fixn['End'] # in microseconds

                list_fixn_dur.append(fixn['Duration'])
                list_fixn_x.append(fixn['LocationX'])
                list_fixn_y.append(fixn['LocationY'])

            # ------- loop end: fixations --------

            task_fixn_dur = 1. * (last_fixn_end - first_fixn_st) * SEC_FACTOR

            fixn_dur_sum = np.nan_to_num(sum(list_fixn_dur))
            fixn_dur_avg = np.nan_to_num(np.mean(list_fixn_dur))
            fixn_dur_sd = np.nan_to_num(np.std(list_fixn_dur))

            fixn_n_l1 = sum(100 <= i < 250 for i in list_fixn_dur)
            fixn_n_l2 = sum(250 <= i < 400 for i in list_fixn_dur)
            fixn_n_l3 = sum(400 <= i < 550 for i in list_fixn_dur)
            fixn_n_l4 = sum(550 <= i for i in list_fixn_dur)

            fixn_hull_area, fixn_hull_x, fixn_hull_y = 0, 0, 0

            if fixn_n >= 3:
                fixn_hull_area, fixn_hull_x, fixn_hull_y \
                    = calc_convex_hull_area_centroid(list_fixn_x, list_fixn_y)

                # normalizing by stimulus area
                fixn_hull_area /= (SCREEN_WIDTH * SCREEN_HEIGHT)

            # how to normalize?
            scan_dist_euclid = calc_scan_dist_euclid(
                list_fixn_x, list_fixn_y,
                normalizer_x=SCREEN_WIDTH, normalizer_y=SCREEN_HEIGHT
            )

            avg_sacc_len = 0

            if fixn_n > 1:
                # no. of saccades = no. fixations - 1
                avg_sacc_len = scan_dist_euclid / (fixn_n - 1)


            # normalizing for stimulus dimensions
            scan_dist_h = calc_scan_dist_hv(list_fixn_x, normalizer=SCREEN_WIDTH)
            scan_dist_v = calc_scan_dist_hv(list_fixn_y, normalizer=SCREEN_HEIGHT)

            scan_hv_ratio = scan_dist_h

            if scan_dist_v > 0:
                scan_hv_ratio = (scan_dist_h / scan_dist_v)
                """
                thus when vertical distance is zero, 
                we assume vertical distance is 1 pixel
                """


            scan_speed = scan_dist_euclid / task_dur
            scan_speed_h = scan_dist_h / task_dur
            scan_speed_v = scan_dist_v / task_dur

            fixn_per_scan_dist = fixn_n

            if scan_dist_euclid != 0:
                fixn_per_scan_dist = fixn_n / scan_dist_euclid
                """
                thus when euclidean distance is zero, 
                we assume euclidean distance is 1 pixel
                """

            ovrl_fixn_vec0_x, ovrl_fixn_vec0_y, ovrl_fixn_vec0_len, ovrl_fixn_vec0_theta_rad, ovrl_fixn_vec0_theta_deg \
                = calc_overall_fixn_vector(list_fixn_x, list_fixn_y, list_fixn_dur)

            ovrl_fixn_vec_x, ovrl_fixn_vec_y, ovrl_fixn_vec_len, ovrl_fixn_vec_theta_rad, ovrl_fixn_vec_theta_deg \
                = calc_overall_fixn_vector(
                list_fixn_x, list_fixn_y, list_fixn_dur, x_c=(SCREEN_WIDTH-1)/2, y_c=(SCREEN_HEIGHT-1)/2,
                normalizer_x=SCREEN_WIDTH, normalizer_y=SCREEN_HEIGHT
            )


            bulk_insert.append([
                int(userid)
                , str(docid)
                , DOCID_USERID
                , p_rlvnc
                , t_rlvnc
                , int(task_st)
                , int(task_end)
                , float(round(task_dur, 10))
                , int(first_fixn_st)
                , int(last_fixn_end)
                , float(round(task_fixn_dur, 10))
                , int(fixn_n)

                , int(fixn_n_l1)
                , int(fixn_n_l2)
                , int(fixn_n_l3)
                , int(fixn_n_l4)

                , float(fixn_rate)
                , float(round(fixn_dur_sum, 10))
                , float(round(fixn_dur_avg, 10))

                , float(round(fixn_dur_sd, 10))

                , float(round(fixn_hull_area, 10))
                , float(round(fixn_hull_x, 10))
                , float(round(fixn_hull_y, 10))
                , float(round(scan_dist_euclid, 10))
                , float(round(scan_dist_h, 10))
                , float(round(scan_dist_v, 10))
                , float(round(scan_hv_ratio, 10))
                , float(round(scan_speed, 10))
                , float(round(scan_speed_h, 10))
                , float(round(scan_speed_v, 10))
                , float(round(fixn_per_scan_dist, 10))

                , float(round(avg_sacc_len, 10))

                , float(round(ovrl_fixn_vec0_x, 10))
                , float(round(ovrl_fixn_vec0_y, 10))
                , float(round(ovrl_fixn_vec0_len, 10))
                , float(round(ovrl_fixn_vec0_theta_rad, 10))
                , float(round(ovrl_fixn_vec0_theta_deg, 10))
                , float(round(ovrl_fixn_vec_x, 10))
                , float(round(ovrl_fixn_vec_y, 10))
                , float(round(ovrl_fixn_vec_len, 10))
                , float(round(ovrl_fixn_vec_theta_rad, 10))
                , float(round(ovrl_fixn_vec_theta_deg, 10))
            ])

            # bulk insert whenever BULK_THRESH rows are ready
            if len(bulk_insert) >= BULK_THRESH:

                df_null_check = pd.DataFrame(bulk_insert)
                num_null = df_null_check.isnull().sum().sum()

                if num_null > 0:
                    print("no. of nulls = %d " % num_null)
                    print(df_null_check.loc[df_null_check.isnull().any(axis=1)].to_string())

                cur_insert.executemany(sql_insert, bulk_insert)
                tot_rows_insert += len(bulk_insert)
                print("total rows inserted: %5d / %5d" % (tot_rows_insert, tot_user_doc))
                bulk_insert = []
            # ------- loop end: user doc --------
            pass

        #############################
        # bulk insert remaining rows
        #############################
        df_null_check = pd.DataFrame(bulk_insert)
        num_null = df_null_check.isnull().sum().sum()

        if num_null > 0:
            print("no. of nulls = %d " % num_null)
            print(df_null_check.loc[df_null_check.isnull().any(axis=1)].to_string())

        cur_insert.executemany(sql_insert, bulk_insert)
        tot_rows_insert += len(bulk_insert)
        print("total rows inserted: %5d" % tot_rows_insert)
        bulk_insert = []


        conn.commit()
        print('Done. Finished')

    finally:

        cur_select_usr_doc.close()
        cur_select_fixns.close()
        cur_insert.close()

        conn.close()
        tunnel.stop()
        tunnel.close()




if __name__ == '__main__':
    main()
    exit(0)