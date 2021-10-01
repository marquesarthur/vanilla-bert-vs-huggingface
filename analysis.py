import json
import unittest
from collections import defaultdict
from colorama import Fore, Style
import numpy as np
import pandas as pd
import logging
import sys

logger = logging.getLogger()


class ReportAnalysis(object):
    
    def __init__(self):
        self.pandas_table = defaultdict(list)
        
        
    def pd_table(self):
        return pd.DataFrame.from_dict(self.pandas_table)
    
    def filter_nan(self, _input):
        return [x for x in _input if str(x) != 'nan']

    def get_fold_metrics(self, input_file, type='overall'):
        fold_results = dict()
        __precision, __recall, __fscore = [], [], []

        with open(input_file) as input_file:
            fold_results = json.load(input_file)

        for key_i, value in fold_results.items():
            if isinstance(value, dict):
                for key_j, _data in value.items():
                    if key_j == type:
                        __precision.append(np.mean(self.filter_nan(_data['precision'])))
                        __recall.append(np.mean(self.filter_nan(_data['recall'])))
                        __fscore.append(np.mean(self.filter_nan(_data['fscore'])))


        return np.mean(self.filter_nan(__precision)), np.mean(self.filter_nan(__recall)), np.mean(self.filter_nan(__fscore))
    
    
    def report_BERT_metrics(self, pattern, source_type='overall', verbose=False):
        bert_file = pattern + '_base.json'
        bert_file_A = pattern + '_fe.json' # for frame-elements filter
        bert_file_B = pattern + '_fa.json' # for frame-association filters   


        # <-------------------------------------------------------------------------------------- BERT    
        bert_precision, bert_recall, bert_f1score = self.get_fold_metrics(bert_file, type=source_type)

        self.pandas_table["technique"].append(bert_file.replace("output/", "").replace("_base.json", ""))
        self.pandas_table["precision"].append(bert_precision)
        self.pandas_table["recall"].append(bert_recall)    
        self.pandas_table["f1-score"].append(bert_f1score)

        if verbose:
            logger.info("")
            logger.info(f"BERT " + Fore.RED + source_type.upper() + Style.RESET_ALL + " metrics")
            logger.info("precision: " + Fore.RED + "{:.3f}".format(bert_precision) + Style.RESET_ALL)
            logger.info("recall:    " + Fore.RED + "{:.3f}".format(bert_recall) + Style.RESET_ALL)
            logger.info("f1-score:  " + Fore.RED + "{:.3f}".format(bert_f1score) + Style.RESET_ALL)

        # <-------------------------------------------------------------------------------------- BERT    
        bert_precision, bert_recall, bert_f1score = self.get_fold_metrics(bert_file_A, type=source_type)

        self.pandas_table["technique"].append(bert_file_A.replace("output/", "").replace("_fe.json", " w/ frame-elements"))
        self.pandas_table["precision"].append(bert_precision)
        self.pandas_table["recall"].append(bert_recall)    
        self.pandas_table["f1-score"].append(bert_f1score)

        if verbose:
            logger.info("")
            logger.info(Fore.RED + "frame-elements" + Style.RESET_ALL + " metrics")
            logger.info("precision: " + Fore.RED + "{:.3f}".format(bert_precision) + Style.RESET_ALL)
            logger.info("recall:    " + Fore.RED + "{:.3f}".format(bert_recall) + Style.RESET_ALL)
            logger.info("f1-score:  " + Fore.RED + "{:.3f}".format(bert_f1score) + Style.RESET_ALL)

        # <-------------------------------------------------------------------------------------- BERT    
        bert_precision, bert_recall, bert_f1score = self.get_fold_metrics(bert_file_B, type=source_type)

        self.pandas_table["technique"].append(bert_file_B.replace("output/", "").replace("_fa.json", " w/ frame-associations"))
        self.pandas_table["precision"].append(bert_precision)
        self.pandas_table["recall"].append(bert_recall)    
        self.pandas_table["f1-score"].append(bert_f1score)

        if verbose:
            logger.info("")
            logger.info(Fore.RED + "frame-associations" + Style.RESET_ALL + " metrics")
            logger.info("precision: " + Fore.RED + "{:.3f}".format(bert_precision) + Style.RESET_ALL)
            logger.info("recall:    " + Fore.RED + "{:.3f}".format(bert_recall) + Style.RESET_ALL)
            logger.info("f1-score:  " + Fore.RED + "{:.3f}".format(bert_f1score) + Style.RESET_ALL)
