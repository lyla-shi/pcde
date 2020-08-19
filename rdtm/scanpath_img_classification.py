# -*- coding: utf-8 -*-
"""
 Author: Li Shi
 Date: 2020-07-07
"""
from __future__ import unicode_literals
# fix random seed for reproducibility
SEED = 15081992 #42 #15081992
import os
os.environ['PYTHONHASHSEED'] = '15081992'                     # <<<<<<<<<<<<<<<< reproducible results
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'               # <<<<<<<<<<<<<<<< reproducible results
import numpy as np
np.random.seed(SEED)                                   # <<<<<<<<<<<<<<<< reproducible results
import random
random.seed(SEED)                                      # <<<<<<<<<<<<<<<< reproducible results
import tensorflow as tf # don't import keras. Use tf.keras
tf.set_random_seed(SEED)                               # <<<<<<<<<<<<<<<< reproducible results


import math, time, sys, socket, pickle
import warnings
from collections import Counter
import matplotlib as mpl
# mpl.rc('font', **{'sans-serif' : 'Arial', 'family' : 'sans-serif'})
from sklearn.utils.class_weight import compute_class_weight


if os.name == 'posix' and os.environ.get('DISPLAY','') == '':
    # print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
from scipy import stats
from scipy.special import logit, expit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, balanced_accuracy_score, \
    precision_recall_curve, average_precision_score, roc_auc_score, accuracy_score, \
    precision_recall_fscore_support
# from tensorflow import keras
"""
# for reproducible results, as suggested here: 
# https://github.com/keras-team/keras/issues/2743#issuecomment-379999712
# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
"""
from tensorflow.python.keras import backend as K                               # <<<<<<<<<<< reproducible results
# sess_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)      # <<<<<<< reproducible results
####Force Tensorflow to use a single thread                                    # <<<<<<<<<<< reproducible results
# sess = tf.Session(graph=tf.get_default_graph(), config=sess_conf)            # <<<<<<<<<<< reproducible results
# K.set_session(sess)                                                          # <<<<<<<<<<< reproducible results
from tensorflow.python.keras.preprocessing.image import img_to_array as img_to_array
from tensorflow.python.keras.preprocessing.image import load_img as load_img
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Model, Sequential, load_model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten
from tensorflow.python.keras.layers import Input, LSTM, Embedding, Dense, concatenate, Dropout
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.regularizers import L1L2
from tensorflow.python.keras.callbacks import EarlyStopping, Callback, ModelCheckpoint, CSVLogger, LambdaCallback
from tensorflow.python.client import device_lib
from functools import partial
from tensorflow.python.keras.applications import \
    DenseNet121, InceptionV3, InceptionResNetV2, NASNetLarge, ResNet50, VGG16, VGG19, Xception, \
    MobileNet, MobileNetV2, NASNetMobile

import matplotlib.pyplot as plt
import cv2
#https://scikit-plot.readthedocs.io/en/stable/
import scikitplot as skplt
from datetime import datetime

# redirect output on remote machines
if 'NB-XPS15-9560' not in socket.getfqdn():
    pass
    # sys.stdout = open("./%s.out" % sys.argv[0], "a")
    # print = partial(print, flush=True)



#############
# parameters
#############

NUM_CLASSES = None #2
MODELS_FOLDER = "./scanpath_img_classify/ml_run_models/"
RESULTS_FOLDER = "./scanpath_img_classify/ml_run_results/"
PLOTS_FOLDER = "./scanpath_img_classify/ml_run_results/"
RUN_OUTPUT_FILE = "./scanpath_img_classify/zz___ML_run_output.csv"

PLOT_FIGSIZE = (8, 8)
PLOT_TOPLOSS_FIGSIZE = (25, 25)
PLOT_TOPLOSS_K = 8

OUTP_COLS = [
    'p_rlvnc',
    # 't_rlvnc',
]

OUTP_COLS_PRETTY = [
    'User Relevance',
    # 'Document Relevance',
]

DICT_LAYER_OUTPUTS = {}
DICT_LAYER_NAMES = {}



# ----- large --------
# 'InceptionV3': [(299, 299),
#     tf.keras.applications.inception_v3.InceptionV3,
#     tf.keras.applications.inception_v3.preprocess_input],
# 'DenseNet169': [(224, 224),
#     tf.keras.applications.densenet.DenseNet169,
#     tf.keras.applications.densenet.preprocess_input],
# 'NASNetLarge': [(331, 331),
#     tf.keras.applications.nasnet.NASNetLarge,
#     tf.keras.applications.nasnet.preprocess_input],
# 'Xception': [(299, 299),
#     tf.keras.applications.xception.Xception,
#     tf.keras.applications.xception.preprocess_input],
# 'NASNetMobile': [(224, 224),
#      tf.keras.applications.nasnet.NASNetMobile,
#      tf.keras.applications.nasnet.preprocess_input],
# ----- mobile --------
# 'MobileNet': [(224, 224),
#     tf.keras.applications.mobilenet.MobileNet,
#     tf.keras.applications.mobilenet.preprocess_input],
# 'MobileNetV2': [(224, 224),
#      tf.keras.applications.mobilenet_v2.MobileNetV2,
#      tf.keras.applications.mobilenet_v2.preprocess_input],
# TODO: squeezent https://cv-tricks.com/tensorflow-tutorial/keras/

P_MODELS = {
    'DenseNet121': [(224, 224),
                    tf.keras.applications.densenet.DenseNet121,
                    tf.keras.applications.densenet.preprocess_input],
    'DenseNet201': [(224, 224),
                    tf.keras.applications.densenet.DenseNet201,
                    tf.keras.applications.densenet.preprocess_input],
    'InceptionResNetV2': [(224, 224),
                          tf.keras.applications.inception_resnet_v2.InceptionResNetV2,
                          tf.keras.applications.inception_resnet_v2.preprocess_input],
    'ResNet50': [(224, 224),
                 tf.keras.applications.resnet50.ResNet50,
                 tf.keras.applications.resnet50.preprocess_input],
    'VGG16': [(224, 224),
              tf.keras.applications.vgg16.VGG16,
              tf.keras.applications.vgg16.preprocess_input],
    'VGG19': [(224, 224),
              tf.keras.applications.vgg19.VGG19,
              tf.keras.applications.vgg19.preprocess_input],
}



M = 'VGG16'

# FREEZE_LAYERS = True
FREEZE_LAYERS = False

# LR = 1e-4
# LR = 1e-5
LR = 1e-6


# REG = L1L2(l1=0.01, l2=0.01)
REG = None

# IMG_VERSION = "v1"
IMG_VERSION = "v2"

BATCH_SIZE = 16 #None #16
EPOCHS = 6 #6 #5 #50  # no. of epochs

IMG_NORMALIZER = 1. #255.
PREPROCESS = True

USE_CUSTOM_IMG_SIZE = False
# USE_CUSTOM_IMG_SIZE = True
CUSTOM_H, CUSTOM_W = 1000, 1000 # H (rows) x W (cols)



H, W = P_MODELS[M][0][0], P_MODELS[M][0][1] # H (rows) x W (cols)
F_MODEL = P_MODELS[M][1]
F_PREPROC = P_MODELS[M][2]

if USE_CUSTOM_IMG_SIZE: H, W = CUSTOM_H, CUSTOM_W

def getNN():

    inp_shape = (H, W, 3)  # new tuple (H x W x n_channels)
    InpImg = Input(shape=inp_shape, name='inp_img')

    # base pre-trained model
    base_model = F_MODEL(weights='imagenet', include_top=False, input_tensor=InpImg)
    x = base_model.output

    # name all custom layers as nb_ to log in file
    x = Flatten(name='nb_flatten')(x)
    # x = GlobalAveragePooling2D(name='nb_avg_pool')(x)
    x = Dense(256, activation='relu', name='nb_fc1', kernel_regularizer=REG)(x)
    x = Dropout(rate=0.2, seed=SEED, name='nb_drop1')(x)
    prediction = Dense(1, activation='sigmoid', name='nb_output')(x)

    model = Model(inputs=base_model.input, outputs=prediction)

    # freeze layers in base_model
    if FREEZE_LAYERS:
        for layer in base_model.layers: #[:10]: #top 10 layers
            layer.trainable = False

    #############
    # GPU code
    #############
    num_gpu = len(get_available_gpus())

    if num_gpu >= 2:
        parallel_model = tf.keras.utils.multi_gpu_model(model, gpus=num_gpu)
        print("Training using %d GPUs" % num_gpu)
    else:
        print("Training using %d GPU or CPU" % num_gpu)
        parallel_model = model

    #################
    # compile model
    #################

    parallel_model.compile(
        #####################
        # fine-tuning should be done with a very slow learning rate, and typically with the
        # SGD optimizer rather than an adaptative learning rate optimizer such as RMSProp.
        # This is to make sure that the magnitude of the updates stays very small,
        # so as not to wreck the previously learned features.
        # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
        #
        # SGD + momentum
        # https://towardsdatascience.com/a-bunch-of-tips-and-tricks-for-training-deep-neural-networks-3ca24c31ddc8
        # https://arxiv.org/abs/1705.08292
        # There are three learning-rate starting points to play with:
        # 1e-1, 1e-3 and 1e-6.
        # Fine-tuning: less than 1e-3 (say 1e-4 or 1e-6).
        # Train from scratch: greater than or equal 1e-3
        ######################


        optimizer=optimizers.SGD(lr=LR, momentum=0.9, nesterov=False), # good for fine tuning pre-trained
        # optimizer='rmsprop', # default for from_scratch #1
        # optimizer='adam', # default for from_scratch #2
        # optimizer=optimizers.Adam(lr=L_R),

        loss='binary_crossentropy',  # binary classification (multilabel also) [1,0,0,1,0,1,1,0,...]
        # loss='categorical_crossentropy', #one-hot multiclass [[1,0,0,0], [0,1,0,0], ...]
        # loss='sparse_categorical_crossentropy',  # integer multiclass [1,3,0,2,4,1,7, ..]

        metrics=['accuracy'],
        # , metrics=['binary_accuracy']
    )


    print(parallel_model.summary())
    print("---- End of model summary ----")

    return parallel_model #, model




# get number of GPUs
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']





















def main():

    # consider using argparse

    if len(sys.argv) < 4:
        print("Specify dataset (pcde / pcde_balanced), mode (train / validation / test), run no. (from excel)")
        print("   [model name to hotstart from]")
        print("   [plot ROC, PR, Conf Matrix and generate files? (0 / 1)]")
        exit(1)

    dataset = sys.argv[1]
    mode = sys.argv[2]
    run_no = int(sys.argv[3])

    hotstart_model_name = None

    if len(sys.argv) >= 5:
        hotstart_model_name = sys.argv[4]

    plot_stats_print_files = False

    if len(sys.argv) >= 6 and sys.argv[5] == "1":
        plot_stats_print_files = True

    # dataset check
    if dataset == 'pcde':
        dataset_pretty = 'PCDE'
    elif dataset == 'pcde_balanced':
        dataset_pretty = 'PCDE - Balanced'
    else:
        print('Invalid dataset: %s. \nValid: (pcde / pcde_balanced)' % dataset)
        exit(1)

    # mode check
    if mode not in ['train', 'validation', 'test']:
        print('Invalid mode: %s. \nValid: (train / validation / test)' % mode)
        exit(1)


    #####################
    # SET ALL PATHS
    #####################

    # GPU server paths
    path_imgs = './scanpath_img_classify/img_all_%s/' % IMG_VERSION

    # model
    trained_model_path = os.path.join(MODELS_FOLDER, 'model__%s__run_%02d.hdf5' % (dataset, run_no))

    # timestamp str, for output files, to allow
    # multiple predictions from the same trained model
    ts = datetime.today().strftime('%Y%m%d_%H%M%S')

    # predictions Excel file
    path_pred_xlsx = os.path.join(RESULTS_FOLDER, 'preds__%s__%s_run_%02d__%s.xlsx' % (dataset, mode, run_no, ts))

    # plots
    path_pr = os.path.join(PLOTS_FOLDER, 'pr__%s__%s_run_%02d__%s.png' % (dataset, mode, run_no, ts))
    path_roc = os.path.join(PLOTS_FOLDER, 'roc__%s__%s_run_%02d__%s.png' % (dataset, mode, run_no, ts))
    path_conf_matrix = os.path.join(PLOTS_FOLDER, 'conf_matrix__%s__%s_run_%02d__%s.png' % (dataset, mode, run_no, ts))
    path_conf_matrix_norm = os.path.join(PLOTS_FOLDER, 'conf_matrix_norm__%s__%s_run_%02d__%s.png' % (dataset, mode, run_no, ts))

    # top losses plots
    path_tpls_selected = os.path.join(PLOTS_FOLDER, 'tpls_sel__%s__%s_run_%02d__%s.png' % (dataset, mode, run_no, ts))
    path_tpls_all = os.path.join(PLOTS_FOLDER, 'tpls_all__%s__%s_run_%02d__%s.png' % (dataset, mode, run_no, ts))
    path_tpls_tp  = os.path.join(PLOTS_FOLDER, 'tpls_tp__%s__%s_run_%02d__%s.png' % (dataset, mode, run_no, ts))
    path_tpls_tn  = os.path.join(PLOTS_FOLDER, 'tpls_tn__%s__%s_run_%02d__%s.png' % (dataset, mode, run_no, ts))
    path_tpls_fp  = os.path.join(PLOTS_FOLDER, 'tpls_fp__%s__%s_run_%02d__%s.png' % (dataset, mode, run_no, ts))
    path_tpls_fn  = os.path.join(PLOTS_FOLDER, 'tpls_fn__%s__%s_run_%02d__%s.png' % (dataset, mode, run_no, ts))


    print('**************************************************************')
    print("  Dataset: %s \t Mode: %s \t Run: %02d" % (dataset, mode, run_no))
    if hotstart_model_name is not None:
        print("  Hot-starting from: %s" % hotstart_model_name)
    print('**************************************************************')

    model = None

    # ---------------------------------- TRAINING --------------------------------

    if mode == 'train':

        # do not proceed if model already exists
        # to prevent overwriting the model
        if os.path.isfile(trained_model_path):
            resp = input('Overwrite model for run: %02d? [Y/N] ' % run_no)

            if resp != "Y":
                exit(1)


        ############
        # get model
        ############



        if hotstart_model_name is None:
            print('Building model...')
            model = getNN()
        else:
            hotstart_model_path = os.path.join(MODELS_FOLDER, hotstart_model_name)

            if os.path.isfile(hotstart_model_path):
                print('Loading hot-start model...')
                model = load_model(hotstart_model_path)
            else:
                print('Model does not exist: %s' % hotstart_model_path)
                exit(1)



        ################
        # training data
        ################
        print('Loading training data...')

        seq_train = ScanpathDataSequence(
            dataset=dataset,
            csv_path='./scanpath_img_classify/%s__labels_train.csv' % dataset,
            img_path=path_imgs,
            mode='train',
        )

        print('Loading validation data...')

        seq_val = ScanpathDataSequence(
            dataset=dataset,
            csv_path='./scanpath_img_classify/%s__labels_val.csv' % dataset,
            img_path=path_imgs,
            mode='validation',
        )


        callback_list = []
        """
        ############
        # callbacks
        ############
        
        reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, mode='auto') #default min_delta = 0.001
        early = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=2, mode='auto')

        callback_list = [
            # csv_logger_batchwise,
            # csv_logger,
            # checkpoint,
            # early,
        ]
        """

        ##############
        # train
        ##############

        print('\n\n----------- Start TRAINING Run %s ------------\n\n' % run_no)

        model.fit_generator(
            generator=seq_train,
            epochs=EPOCHS,
            # validation_data=seq_val,
            # use_multiprocessing=True,
            # workers=4 #len(get_available_gpus()),
            verbose=1,
            callbacks=callback_list,
            # class_weight=dict_class_wt,
            shuffle=True
        )

        print("Saving model...")
        model.save(trained_model_path)








    # ---------------------------------- PREDICTION --------------------------------

    if mode in ['validation', 'test', 'train']:
        #'train' => to do validation right after training


        #check if model exists
        if not os.path.isfile(trained_model_path):
            print('Model does not exist for this run: %02d' % run_no)
            print('Please run training first (mode: train)')
            exit(1)

        if mode in ['validation', 'train']:
            gt_file = './scanpath_img_classify/%s__labels_val.csv' % dataset
        elif mode == 'test':
            #gt_file = './scanpath_img_classify/%s__labels_test.csv' % dataset
            gt_file = './scanpath_img_classify/%s__labels_val.csv' % dataset

        ############
        # LOAD DATA
        ############

        print("Loading prediction data...")

        seq_test = ScanpathDataSequence(
            dataset=dataset,
            csv_path=gt_file,
            img_path=path_imgs,
            mode='test',
        )

        ##########
        # PREDICT
        ##########

        if mode != 'train' or model is None:
            print("Loading model...")
            model = load_model(trained_model_path)

        print('\n\n-.-.-.-.-.-.-.-.- Start PREDICTION Run %s -.-.-.-.-.-.-.-.-\n\n' % run_no)
        y_pred = model.predict_generator(generator=seq_test, verbose=1)

        y_pred = np.array(y_pred)

        ####################
        # LOAD GROUND TRUTH
        ####################

        print("Loading ground truth...")

        gt_data = pd.read_csv(gt_file, skipinitialspace=True, engine='python')
        y_true = np.array(gt_data[OUTP_COLS].values)

        DOCID_USERIDs = gt_data['DOCID_USERID'].values

        # sanity check
        if len(y_true) != len(y_pred):
            print('ERROR: Ground truth vs. prediction lengths mismatch!!!')
            exit(1)

        ##############################################
        # CALCULATING PREDICTION LOSS FOR EACH SAMPLE
        ##############################################
        eps = 1e-15 # to prevent log(0) error
        p = np.clip(np.nan_to_num(y_pred), eps, 1 - eps)
        y_loss = -( y_true * np.log(p) + (1 - y_true) * np.log(1 - p))




        ##################################
        # save excel file of predictions
        ##################################

        print("Creating Excel file of predictions...")

        df_xlsx = pd.DataFrame(DOCID_USERIDs)
        df_xlsx.columns = ["DOCID_USERID"]

        df_xlsx.insert(loc=len(df_xlsx.columns), column='y_true', value=y_true)
        df_xlsx.insert(loc=len(df_xlsx.columns), column='y_pred', value=y_pred)
        df_xlsx.insert(loc=len(df_xlsx.columns), column='y_loss', value=y_loss)

        df_false_positive = \
            df_xlsx.loc[(df_xlsx['y_true'] == 0) & (df_xlsx['y_pred'] > 0.5)]

        df_false_negative = \
            df_xlsx.loc[(df_xlsx['y_true'] == 1) & (df_xlsx['y_pred'] <= 0.5)]


        # if plot_stats_print_files:
        if True:
            writer = pd.ExcelWriter(path_pred_xlsx)
            df_xlsx.to_excel(writer, 'Output')
            df_false_positive.to_excel(writer, 'False Positives')
            df_false_negative.to_excel(writer, 'False Negatives')
            writer.save()
            writer.close()

        ##########################################
        # Print out the metrics (for easy entry
        # in ML Runs Spreadsheet)
        #  - false positives %
        #  - false negatives %
        #  - PR curve AUC
        #  - ROC Auc
        #  - accuracy
        #  - precision
        #  - recall
        #  - F1 score
        ##########################################

        df = df_xlsx

        """
         - ROC curves are appropriate when the observations are balanced 
           between each class, whereas
           
         - Precision-recall curves are appropriate for imbalanced datasets.
        """

        fp_pct = 1. * sum((df['y_true'] == 0) & (df['y_pred'] > 0.5)) / sum(df['y_true'] == 0)
        fn_pct = 1. * sum((df['y_true'] == 1) & (df['y_pred'] <= 0.5)) / sum(df['y_true'] == 1)

        roc_auc = roc_auc_score(y_true, y_pred)

        acc = accuracy_score(y_true, y_pred.round().astype(int))
        acc_balncd = balanced_accuracy_score(y_true, y_pred.round().astype(int))

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred.round().astype(int), average='binary')

        ap = average_precision_score(y_true, y_pred.round().astype(int))


        # appending to file
        run_outp_file_exists = os.path.exists(RUN_OUTPUT_FILE)

        reg_str = ""
        if REG is not None: reg_str = "L1L2(.01, .01)"

        run_results = [
            ('TimeStamp', ts),
            ('Run', run_no),
            ('Dataset', dataset),
            ('img_version', IMG_VERSION),
            ('Mode', mode),

            ('img_size', f'({H} x {W})'),
            ('Model', M),
            ('Frozen?', str(FREEZE_LAYERS)),
            ('lr', LR),
            ('Reg',  reg_str),

            ('batch_size', BATCH_SIZE),
            ('epochs', EPOCHS),

            ('FP', '%.3f' % fp_pct),
            ('FN', '%.3f' % fn_pct),
            ('ROC_AUC', '%.3f' % roc_auc),
            ('acc', '%.3f' % acc),
            ('acc_balncd', '%.3f' % acc_balncd),
            ('prec', '%.4f' % precision),
            ('recall', '%.4f' % recall),
            ('AP', '%.6f' % ap),
            ('F1', '%.6f' % f1),
        ]

        str_header = ""
        str_body = ""

        for item in run_results:
            str_header += (str(item[0]) + ",")
            str_body += (str(item[1]) + ",")

        with open(RUN_OUTPUT_FILE, 'a+') as f:
            if not run_outp_file_exists: f.write(str_header + "\n")
            f.write(str_body + "\n")
            f.close()


        # printing
        print("\n\nResults:")
        print("FP\tFN\tROC_AUC\tacc\tacc_bal\tprec\trecl\tap\tf1")

        print("%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" % (
            fp_pct, fn_pct, roc_auc,
            acc, acc_balncd, precision, recall, ap, f1
        ))

        ##############
        # TOP LOSSES
        ##############

        print("\n\nPlotting Top Losses")

        class_labels = {
            0: 'Irrel',
            1: 'Rel'
        }

        print("selected")
        plot_top_losses(PLOT_TOPLOSS_K, df_xlsx, run_no, path_imgs,
                        path_tpls_selected, mode=None,
                        class_labels=class_labels,
                        figsize=PLOT_TOPLOSS_FIGSIZE,
                        model=model, plot_heatmap=True,
                        lst_docid_userid=[
                            'NYT19991102.0265_7',
                            'NYT20000110.0271_9',
                            'NYT19980611.0085_9',
                            'NYT19990108.0267_6',
                            'NYT19981107.0047_12',
                            'NYT19990901.0251_4',

                            'APW19981124.1419_8',
                            'NYT19991021.0297_17',
                            'NYT20000214.0441_9',
                            'XIE19980208.0052_4',
                            'NYT20000110.0271_8',
                            'APW19990722.0257_22',

                        ])

        print("all")
        plot_top_losses(PLOT_TOPLOSS_K, df_xlsx, run_no, path_imgs,
                        path_tpls_all, mode=None,
                        class_labels=class_labels,
                        figsize=PLOT_TOPLOSS_FIGSIZE,
                        model=model, plot_heatmap=True,
                        )

        """
        print("tp")
        plot_top_losses(PLOT_TOPLOSS_K, df_xlsx, run_no, path_imgs,
                        path_tpls_tp, mode='tp',
                        class_labels=class_labels,
                        figsize=PLOT_TOPLOSS_FIGSIZE,
                        model=model, plot_heatmap=True,
                        )

        print("tn")
        plot_top_losses(PLOT_TOPLOSS_K, df_xlsx, run_no, path_imgs,
                        path_tpls_tn, mode='tn',
                        class_labels=class_labels,
                        figsize=PLOT_TOPLOSS_FIGSIZE,
                        model=model, plot_heatmap=True,
                        )
        
        print("fp")
        plot_top_losses(PLOT_TOPLOSS_K, df_xlsx, run_no, path_imgs,
                        path_tpls_fp, mode='fp',
                        class_labels=class_labels,
                        figsize=PLOT_TOPLOSS_FIGSIZE,
                        model=model, plot_heatmap=True,
                        )

        print("fn")
        plot_top_losses(PLOT_TOPLOSS_K, df_xlsx, run_no, path_imgs,
                        path_tpls_fn, mode='fn',
                        class_labels=class_labels,
                        figsize=PLOT_TOPLOSS_FIGSIZE,
                        model=model, plot_heatmap=True,
                        )
        """


        if plot_stats_print_files:
            ##########################################################
            # PERFORMANCE PLOTS
            # For plots
            # y_true (array-like, shape (n_samples))
            # y_probas (array-like, shape (n_samples, n_classes))
            # https://scikit-plot.readthedocs.io/en/stable/
            ##########################################################

            y_true_plot = np.copy(y_true)
            y_true_plot.shape = (len(y_true))

            y_pred_plot = np.zeros((len(y_true), 2)) #np.copy(y_pred)
            y_pred_plot_cm = np.zeros(len(y_true))

            for i in range(len(y_pred)):
                y_pred_plot[i][0] = 1.0 - y_pred[i]
                y_pred_plot[i][1] = y_pred[i]

                y_pred_plot_cm[i] = int(y_pred[i] > 0.5)


            print("Plotting Performance Plots")

            #########################
            # Precision Recall Plot
            #########################
            plt.figure()
            skplt.metrics.plot_precision_recall(
                y_true_plot, y_pred_plot,
                plot_micro=False,
                figsize=PLOT_FIGSIZE,
                title="Precision-Recall Curves (Run %02d)" % run_no,
            )
            plt.savefig(path_pr)#, bbox_inches='tight', pad_inches=0)

            ###############
            # ROC AUC Plot
            ###############
            plt.figure()
            skplt.metrics.plot_roc(
                y_true_plot, y_pred_plot,
                plot_micro=False, plot_macro=False,
                figsize=PLOT_FIGSIZE,
                title="ROC Curves (Run %02d)" % run_no,
            )
            plt.savefig(path_roc)#, bbox_inches='tight', pad_inches=0)


            ###################
            # Confusion Matrix
            ###################

            plt.figure()
            skplt.metrics.plot_confusion_matrix(
                y_true_plot, y_pred_plot_cm,
                figsize=PLOT_FIGSIZE,
                title="Confusion Matrix (Run %02d)" % run_no,
            )
            plt.savefig(path_conf_matrix)#, bbox_inches='tight', pad_inches=0)


            ##############################
            # Confusion Matrix Normalized
            ##############################
            plt.figure()
            skplt.metrics.plot_confusion_matrix(
                y_true_plot, y_pred_plot_cm,
                normalize=True,
                figsize=PLOT_FIGSIZE,
                title="Normalized Confusion Matrix (Run %02d)" % run_no,
                cmap='Greens'
            )
            plt.savefig(path_conf_matrix_norm)#, bbox_inches='tight', pad_inches=0)



        print('Finished')






    else:
        print('Invalid mode')
        exit(1)














#########################################################
# ------------- UTILITY CODE ----------------
#########################################################









# similar to fast.ai function
# https://docs.fast.ai/vision.learner.html#_cl_int_plot_top_losses
def plot_top_losses(k, df, run_no, path_imgs, path_save,
                    mode:str=None,  # None / tp / tn / fp / fn
                    class_labels:dict=None,
                    plot_heatmap:bool=False,
                    model=None,
                    figsize=(20, 20),
                    heatmap_thresh:int=16, alpha:float=0.6,
                    print_extra_file:bool=False,
                    cmap:str='magma',  #'jet,
                    lst_docid_userid:list=[]  #if we want to plot only specific DOCID_USERIDs
                    ):

    df1 = df.copy(deep=True)
    mode_pretty = ""

    if mode is not None:
        if mode == "tp":
            df1 = df1.loc[(df['y_true'] == 1) & (df1['y_pred'] > 0.5)]
            mode_pretty = "\nfor True Positives - Correct Relevance Predictions"
        elif mode == "tn":
            df1 = df1.loc[(df['y_true'] == 0) & (df1['y_pred'] <= 0.5)]
            mode_pretty = "\nfor True Negatives - Correct Non-relevance Predictions"
        elif mode == "fp":
            df1 = df1.loc[(df['y_true'] == 0) & (df1['y_pred'] > 0.5)]
            mode_pretty = "\nfor False Positives - Wrong Relevance Predictions"
        elif mode == "fn":
            df1 = df1.loc[(df['y_true'] == 1) & (df1['y_pred'] <= 0.5)]
            mode_pretty = "\nfor False Negatives - Wrong Non-relevance Predictions"
        else:
            print("plot_top_losses(): Invalid mode. Showing global")



    df2 = df1.copy(deep=True)
    df1 = df1.sort_values(by='y_loss', ascending=False) # highest losses => least confident?
    df2 = df2.sort_values(by='y_loss', ascending=True) # lowest losses => most confident?

    df1 = df1.head(k)
    df2 = df2.head(k)

    df_joined = pd.concat([df1, df2])

    #k *= 2 # for showing descending and ascending
    cols = math.ceil(math.sqrt(k*2))
    rows = math.ceil(k*2 / cols) # for most and least confident

    plot_title = '%s: Top-%d Worst and Best Losses, in DESC(▼) and ASC(▲) order (Run %s) %s ' \
                 '\nPrediction / Actual / Loss / Prediction Confidence ' % (
                     M, k, run_no, mode_pretty)

    ###################################
    # plotting specific docid_userids
    ##################################
    if len(lst_docid_userid) > 0:
        # selecting those DOCID_USERIDs
        df_joined = df[df['DOCID_USERID'].isin(lst_docid_userid)]

        # sorting according to list order
        df_joined = df_joined.copy(deep=True)
        df_joined['DOCID_USERID_idx'] = df_joined['DOCID_USERID']
        df_joined = df_joined.set_index('DOCID_USERID_idx')
        df_joined = df_joined.loc[lst_docid_userid]

        k = len(df_joined.index)
        cols = math.ceil(math.sqrt(k))
        rows = math.ceil(k / cols) # for most and least confident
        plot_title = '%s: Selected Doc-User Scanpaths (Run %s)' \
                     '\nPrediction / Actual / Loss / Prediction Confidence ' % (
                         M, run_no, )


    # Create subplots and hide ticks
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    for ax in axes.flat: ax.set(xticks=[], yticks=[])
    fig.suptitle(plot_title, weight='bold', size=18)


    # plot_heatmap part 1
    if plot_heatmap:
        grad_fn = grad_cam_init(model)


    #################
    # LOOP OVER DF
    #################

    m_desc, m_asc = "▼", "▲"
    # m_desc, m_asc = "↓", "↑"

    if print_extra_file:
        f_top_loss = open(path_save + '.txt', 'a+')
    i, i_col = 0, 0

    for index, row in df_joined.iterrows():

        docid_userid = row['DOCID_USERID']
        y_pred_prob, y_true, y_loss = row['y_pred'], row['y_true'], row['y_loss']
        y_pred = round(y_pred_prob)

        if y_pred == 0:
            y_pred_prob = 1-y_pred

        #####################################################
        # writing docid userid to file for easy copy/paste
        ####################################################
        if print_extra_file:
            f_top_loss.write(docid_userid + ",")
        i_col += 1

        if i_col >= cols:
            if print_extra_file:
                f_top_loss.write('\n')
            i_col = 0

        if class_labels is not None:
            y_true = class_labels.get(y_true)
            y_pred = class_labels.get(y_pred)

        #######################################################
        # subplot title for sort order and correct prediction
        #######################################################

        sort_mkr = m_desc
        if i+1 > k: # 2nd half is ascending
            sort_mkr = m_asc

        if len(lst_docid_userid) > 0: sort_mkr = ""

        subplot_title = u'%s / %s / %.2f %s / %.1f%%' % (y_pred, y_true, y_loss, sort_mkr, y_pred_prob*100)

        pred_conf_color = "green"

        if y_true != y_pred:
            pred_conf_color = "red"

        ###########
        # plotting
        ###########
        img_path = os.path.join(path_imgs, docid_userid + '.png')
        img = plt.imread(img_path)
        h, w = img.shape[0], img.shape[1]

        axes.flat[i].imshow(img)
        axes.flat[i].set_title(subplot_title, size=18, color=pred_conf_color)
        axes.flat[i].set_xlabel(f'{docid_userid}', size=18)

        if plot_heatmap:
            cam = grad_cam(grad_fn, img_path)
            cam = cv2.resize(cam, (w, h), cv2.INTER_LINEAR)
            axes.flat[i].imshow(cam, cmap=cmap, alpha=alpha)


        i += 1
    # ----------- end df1 loop -----------


    # removing space between title and plots
    # https://stackoverflow.com/a/39334324
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.savefig(path_save, bbox_inches='tight', pad_inches=0.8)

    if print_extra_file:
        f_top_loss.close()



##########################################
# to identify last Conv layer for GradCAM
##########################################
def find_all_layers_recursive(model, layer_type:str=None, level_str:str= ""):
    for i_layer, layer in enumerate(model.layers):
        key = '%s%06d_' % (level_str, i_layer)
        if layer_type is None:
            DICT_LAYER_NAMES[key] = layer.name #type(layer).__name__
            DICT_LAYER_OUTPUTS[key] = layer.output
        else:
            if type(layer).__name__ == layer_type:
                DICT_LAYER_NAMES[key] = layer.name #type(layer).__name__
                DICT_LAYER_OUTPUTS[key] = layer.output

        if hasattr(layer, 'layers') and type(layer.layers) == list:
            find_all_layers_recursive(layer, layer_type, key)


def normalize(x):
    """Utility function to normalize a tensor by its L2 norm"""
    return (x + 1e-10) / (K.sqrt(K.mean(K.square(x))) + 1e-10)


"""
adapted from:
1. F-chollet Book pp 197 (printed 175)
2. https://www.hackevolve.com/where-cnn-is-looking-grad-cam/
3. https://github.com/eclique/keras-gradcam/blob/master/grad_cam.py
broken into 2 parts for optimization
"""

#####################################
# Grad CAM initializer - called once
#####################################
def grad_cam_init(model):


    # model_output = model.output #[0, cls]
    model_output = model.output[:,0]
    # model_output = model.output[0]
    # model_output = model.output[0,0]
    """ All 4 lines above produce same result"""


    DICT_LAYER_NAMES.clear()
    DICT_LAYER_OUTPUTS.clear()
    find_all_layers_recursive(model, 'Conv2D')

    """output of last convolution layer"""
    print("Using output of Conv2d Layer = %s" % DICT_LAYER_NAMES.get(max(DICT_LAYER_NAMES)))
    conv_output = DICT_LAYER_OUTPUTS.get(max(DICT_LAYER_NAMES))

    """Chollet book and eqlique produce identical result"""

    """
    Gradient of the positive class w.r.t.
    the output feature map of last_conv_layer
    """
    grads = K.gradients(model_output, conv_output)[0]

    """
    Vector where each entry is the mean intensity
    of the gradient over a specific feature-map channel
    done by np.mean() later
    """
    # grads = K.mean(grads, axis=(0, 1, 2)) # <<<< chollet book;

    ## Normalize if necessary
    # grads = normalize(grads)
    """No difference in result"""

    """
    Lets you access the values of the quantities
    you just defined: (pooled) grads and the
    output feature map of last_conv_layer, given
    a sample image
    """
    # iterate_gradient_function = K.function([model.input], [conv_output[0], grads]) # <<<< chollet book
    iterate_gradient_function = K.function([model.input], [conv_output, grads])
    return iterate_gradient_function


#########################################
# Grad CAM main - called for each image
#########################################
def grad_cam(iterate_grad_fn, img_path):

    img_arr = img_to_array(load_img(img_path, target_size=(H, W)))
    X = img_arr / IMG_NORMALIZER
    X = np.expand_dims(X, axis=0)
    if PREPROCESS: X = F_PREPROC(X)

    """ Running the fn() for the given sample image """
    conv_outp_val, grads_val = iterate_grad_fn([X])
    conv_outp_val, grads_val = conv_outp_val[0, :], grads_val[0, :, :, :]

    """ done by np.dot() later """
    # for i in range(512): conv_outp_val[:, :, i] *= grads_val[i] # <<<< chollet book;
    # cam = np.mean(conv_outp_val, axis=-1) # <<<< chollet book;

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(conv_outp_val, weights)

    """ Process CAM """
    cam = np.maximum(cam, 0)
    cam_max = cam.max()
    if cam_max != 0: cam /= cam_max

    return cam









# generates batches for train / validation / test
class ScanpathDataSequence(tf.keras.utils.Sequence):
    """
    Custom Sequence object to train a model on out-of-memory datasets.
    https://blog.ml6.eu/training-and-serving-ml-models-with-tf-keras-3d29b41e066c
    """

    def __init__(self, dataset, csv_path, img_path, mode='train'):
        """
        dataset: pcde / pcde_balanced
        csv_path: path to a .csv file that contains columns with image names and labels
        img_path: path to images
        im_size_tup: image size (H x W)
        mode: when in training mode, data will be shuffled between epochs
        """
        self.df = pd.read_csv(csv_path, skipinitialspace=True, engine='python')
        self.mode = mode
        self.dataset = dataset


        # image list
        self.image_list = self.df['IMG'].apply(
            lambda x: os.path.join(img_path, x)
        ).tolist()

        # outputs
        self.y_lbl = self.df[OUTP_COLS].values

        """
        self.y_lbl_combined = []

        for row in self.y_lbl:
            self.y_lbl_combined.append(
                int(str(row)[1:-1].replace(' ', ''), 2)
            )

        self.y_lbl_combined = np.array(self.y_lbl_combined)
        """


    def __len__(self):
        return int(math.ceil(len(self.df) / float(BATCH_SIZE)))


    def on_epoch_end(self):
        # Shuffles indexes after each epoch
        self.indexes = range(len(self.image_list))
        if self.mode == 'train':
            self.indexes = random.sample(self.indexes, k=len(self.indexes))


    def get_batch_x(self, idx):
        # Fetch a batch of training data
        batch_imgs = self.image_list[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE]
        X = np.array([
            img_to_array(load_img(im, target_size=(H, W))) / IMG_NORMALIZER
            for im in batch_imgs
        ])
        if PREPROCESS: X = F_PREPROC(X)
        return X


    def get_batch_y(self, idx):
        # Fetch a batch of labels
        # return_val = self.y_lbl_combined[idx * self.batch_size: (idx + 1) * self.batch_size]
        y = self.y_lbl[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE, :]
        return y


    def __getitem__(self, idx):
        batch_x = self.get_batch_x(idx)
        batch_y = self.get_batch_y(idx)
        return batch_x, batch_y


    """
    how is this used ?
    """
    def get_class_weights(self):
        ############################################
        # create class weights dict (for imbalance)
        ############################################

        dict_class_weights = dict()

        if self.mode == 'train':
            list_labels = list(np.nonzero(self.y_lbl)[1])  # taking the col positions as labels

            list_class_weights = compute_class_weight('balanced', np.unique(list_labels), list_labels)
            dict_class_weights = dict(enumerate(list_class_weights))

        return dict_class_weights













#########################################################
# ------------- HALF BAKED CODE ----------------
#########################################################




# code for comparing two models
def model_compare():
    ########################################################
    # check with run 01 model
    #
    # model.get_weights() returns a list of all
    #     weight tensors in the model, as Numpy arrays
    #
    # get_config() returns a dictionary containing the
    #     configuration of the model (with weights)
    ########################################################

    model_base_path = 'D:/ALL_WORKSPACES/R&D/IX_Lab_Repo/PCDE/scanpath_img_classify/ml_run_models/'
    m1_path = os.path.join(model_base_path, 'good_model_from_run_01.hdf5')
    m2_path = os.path.join(model_base_path, 'model__pcde_balanced__run_04.hdf5')

    m1, m2 = load_model(m1_path), load_model(m2_path)
    d1, d2 = m1.get_config(), m2.get_config()
    w1, w2 = m1.get_weights(), m2.get_weights()

    print(d1 == d2)

    print(len(m1.layers) == len(m2.layers))

    # comparing weights and biases, layer by layer
    # first_layer_weights = model.layers[0].get_weights()[0]
    # first_layer_biases  = model.layers[0].get_weights()[1]
    # second_layer_weights = model.layers[1].get_weights()[0]
    # second_layer_biases  = model.layers[1].get_weights()[1]
    # https://stackoverflow.com/a/44569375

    n_tot_w = 0
    n_tot_w_diff = 0

    for i in range(len(m1.layers)):
        l1 = len(m1.layers[i].get_weights())
        l2 = len(m2.layers[i].get_weights())

        if l1 != l2:
            print("Layer %s weight lengths don't match" % i)
        else:
            print("Layer %s\tlength: %s" % (i, l1))

            for j in range(l1):
                w1_ij = m1.layers[i].get_weights()[j]
                w2_ij = m2.layers[i].get_weights()[j]

                if np.array_equal(w1_ij, w2_ij):
                    print("    weights[%s] OK" % (j))
                else:
                    if w1_ij.shape != w2_ij.shape: # shape difference?
                        print("    weights[%s]: shapes differ" % (j),
                              w1_ij.shape, "\t", w2_ij.shape
                              )
                    else: #num different elements
                        n_ij_diff = np.sum(w1_ij == w2_ij)
                        n_tot_w_diff += n_ij_diff

                        n_ij_tot = w1_ij.size
                        n_tot_w += n_ij_tot

                        pct_ij_diff = n_ij_diff / n_ij_tot * 100
                        print("    weights[%s]: elements differ [%8d / %8d] (%02.2f %%)"
                              % (j, n_ij_diff, n_ij_tot, pct_ij_diff)
                              )

    pct_tot_diff = n_tot_w_diff / n_tot_w * 100
    print("\nTotal count of weights different [%12d / %12d] (%02.2f %%)"
          % (n_tot_w_diff, n_tot_w, pct_tot_diff)
          )


    print(len(w1) == len(w2))

    for i in range(len(w1)):
        w1_i, w2_i = w1[i], w2[i]
        print("Layer", i, np.array_equal(w1_i, w2_i))


    # architecture compare
    # for l1, l2 in zip(m1.layers, m2.layers):
    #     print(l1.get_config() == l2.get_config())

    # print(m1.get_config() == m2.get_config())



    pass















#########################################################
# ------------- MAIN ----------------
#########################################################


if __name__ == "__main__":
    main()
    exit(0)