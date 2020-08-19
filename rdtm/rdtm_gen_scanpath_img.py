"""
 Author: Li Shi
 Date: 2020-06-30
"""
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.pylab as pl
from sshtunnel import SSHTunnelForwarder
import pymysql
import sys, os, socket
from functools import partial

# redirect output on remote machines
# if 'NB-XPS15-9560' not in socket.getfqdn():
#     pass
#     sys.stdout = open("./%s.out" % sys.argv[0], "a")
#     print = partial(print, flush=True)

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

# https://matplotlib.org/3.1.1/api/markers_api.html#module-matplotlib.markers
FIXN_DUR_LEVEL_MARKER = {
    0: "o",
    1: ".", #level 1
    2: "*", #level 2
    3: "p", #level 3
    4: "X", #level 4
}

COLOR_LIST = [
    '#ffffff',
    '#ff0000',
    '#ff00ff',
    '#ffff00',
    '#ffffff'
]

PRINT_THRESH = 20
X_MAX = 1024
Y_MAX = 768
DPI = 96
VIZ_PLOT_PATH_BASE = "./scanpath_img/"
DF_COLS = [
    'x',
    'y',
    'fixn_dur',

    'fixn_dur_lvl', # 1,2,3,4,5 --> like CHIIR'20 paper
    # 'Diff_Temporal', # time difference between Start of this fixation and End of previous fixation (saccade duration) (ms)
    # 'Diff_Spatial', # Euclidean distance between this fixation and previous fixation (saccade length)
    # 'Speed', # Diff_Spatial / Diff_Temporal (px / ms)
]


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

    sql_select_usr_doc_viewnum = """
        select *
        from sandbox.rdtm_metrics_user_doc_viewnum a
        order by a.scanpath_id 
    """

    sql_select_fixns = """
         select a.* 
         from sandbox.rdtm_fixations a 
         where a.scanpath_id = %s 
         order by a.fixn_n
    """

    cur_select_usr_doc_viewnum = conn.cursor()
    cur_select_fixns = conn.cursor()

    try:
        cur_select_usr_doc_viewnum.execute(sql_select_usr_doc_viewnum)
        tot_user_doc = int(cur_select_usr_doc_viewnum.rowcount)
        i_user_doc = 0

        ################
        # user-doc loop
        ################
        for usr_doc in cur_select_usr_doc_viewnum:
            scanpath_id = usr_doc['scanpath_id']

            i_user_doc += 1

            if i_user_doc % PRINT_THRESH == 0:
                print('scanpath_id: %s\t'
                      '[%5d / %5d]'
                      % (scanpath_id, i_user_doc, tot_user_doc))

            cur_select_fixns.execute(sql_select_fixns, (scanpath_id))

            ####################################
            # fixations loop - create dataframe
            ####################################
            df = []
            prev_fixn_end = 0
            prev_x = 0
            prev_y = 0
            i_fixn = 0
            line_segments = []
            line_segment_widths = []

            for row_fixations in cur_select_fixns:
                i_fixn += 1

                x = row_fixations['x']
                y = row_fixations['y']
                fixn_dur = row_fixations['fixn_dur']

                # fixn_st = row_fixations['Start']
                # fixn_end = row_fixations['End']

                duration_level = 0
                # diff_temporal = 0
                # diff_spatial = 0
                # speed = 0

                #################
                # Duration_Level
                #################

                if fixn_dur >= 550:
                    duration_level = 4
                elif fixn_dur >= 400:
                    duration_level = 3
                elif fixn_dur >= 250:
                    duration_level = 2
                elif fixn_dur >= 100:
                    duration_level = 1

                ###############
                # Diff Columns
                ###############

                if i_fixn > 1:

                    line_segments.append([
                        (prev_x, prev_y), (x, y)
                    ])


                    # diff_temporal = (fixn_st - prev_fixn_end) / 1000
                    #
                    # diff_spatial = math.sqrt(
                    #     (x - prev_x) ** 2 + (y - prev_y) ** 2
                    # )
                    #
                    # speed = diff_spatial / diff_temporal
                    #
                    # line_segment_widths.append(
                    #     math.log2(1 + diff_temporal)
                    # )

                ###################
                # making dataframe
                ###################
                df.append((
                    x,
                    y,
                    fixn_dur,
                    duration_level,
                    # diff_temporal,
                    # diff_spatial,
                    # speed
                ))

                # prev_fixn_end = fixn_end
                prev_x = x
                prev_y = y

            # ------- loop end: fixation --------

            df = pd.DataFrame(df, columns=tuple(DF_COLS))


            ########
            # plot
            ########
            # plot_save_path = '%s/%s/%s' % (VIZ_PLOT_PATH_BASE, split, p_rlvnc)
            # os.makedirs(plot_save_path, exist_ok=True)

            fig = plt.figure(figsize=(1600/DPI, 1600/DPI), dpi=DPI)
            plt.style.use('dark_background')

            """
            lwidths = np.log2(1 + df['Diff_Temporal']) + 1
            points = np.array([df['LocationX'], df['LocationY']]).T.reshape(-1, 1, 2)

            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            colors = pl.cm.plasma(np.linspace(0, 1, df.shape[0]))
            
            lc = LineCollection(
                    segments,
                    linewidths=lwidths,
                    colors=colors
                )

            plt.gca().add_collection(lc)
            """


            #################
            # plotting lines
            #################
            # https://matplotlib.org/examples/color/colormaps_reference.html
            # should be in contrast with marker colour
            colors = pl.cm.winter(np.linspace(0, 1, df.shape[0]))

            plt.gca().add_collection(
                LineCollection(
                    line_segments,
                    linewidths=2,
                    colors=colors
                ))


            ###################
            # plotting points
            ###################
            # plotting each duration level as a separate scatterplot
            for i_fixn_dur_lvl, i_marker in FIXN_DUR_LEVEL_MARKER.items():

                plt.scatter(
                    df[df['fixn_dur_lvl'] == i_fixn_dur_lvl]['x'],
                    df[df['fixn_dur_lvl'] == i_fixn_dur_lvl]['y'],
                    #alphascalar, optional, default: None
                    #The alpha blending value, between 0 (transparent) and 1 (opaque).
                    alpha=1,
                    #markerMarkerStyle, optional
                    #The marker style. marker can be either an instance of the class or the text shorthand for a particular marker.
                    #Defaults to None, in which case it takes the value of rcParams["scatter.marker"] (default: 'o') = 'o'.
                    marker=i_marker,
                    #s  scalar or array-like, shape (n, ), optional
                    #The marker size in points**2. Default is rcParams['lines.markersize'] ** 2.
                    s=20*i_fixn_dur_lvl*4**2, #i_fixn_dur_lvl, # marker size
                    # c=df[df['Duration_Level'] == i_fixn_dur_lvl]['Duration'], # colour
                    # cmap='hsv',
                    c=COLOR_LIST[i_fixn_dur_lvl],
                    zorder=10 #z-index
                )

            plt.xlim(0, X_MAX)
            plt.ylim(0, Y_MAX)
            plt.axis('off')
            ax = plt.gca()
            ax.invert_yaxis()
            # plt.savefig('%s/%s_%s.png' % (plot_save_path, docid, userid), bbox_inches='tight')
            plt.savefig('%s/rdtm_%s.png' % (VIZ_PLOT_PATH_BASE, scanpath_id), bbox_inches='tight')
            #clear figure
            plt.clf()
            plt.close()


            pass
        # ------- loop end: usr_doc ------------

        pass



        print('Done. Finished')

    finally:

        cur_select_usr_doc_viewnum.close()
        cur_select_fixns.close()

        conn.close()
        tunnel.stop()
        tunnel.close()




if __name__ == '__main__':
    main()
    exit(0)