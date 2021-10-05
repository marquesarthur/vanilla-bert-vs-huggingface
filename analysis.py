import json
import unittest
from collections import defaultdict
from colorama import Fore, Style
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import logging
import sys
from numpy import mean
from numpy import var
from math import sqrt
import scipy
import scipy.stats


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
    
    def get_ir_metrics(self, input_file, type='overall'):
        __precision, __recall, __fscore = [], [], []
        with open(input_file) as input_file:
            fold_results = json.load(input_file)

        for key_i, value in fold_results.items():
            if isinstance(value, dict):
                if key_i == type:
                    __precision += value['precision']
                    __recall += value['recall']
                    __fscore  += value['fscore']


        return np.mean(self.filter_nan(__precision)), np.mean(self.filter_nan(__recall)), np.mean(self.filter_nan(__fscore))
    
    
    def get_fold_metrics_to_list(self, input_file, type='overall'):
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


        return self.filter_nan(__precision), self.filter_nan(__recall), self.filter_nan(__fscore)
    
    
    def get_ir_metrics_to_list(self, input_file, type='overall'):
        __precision, __recall, __fscore = [], [], []
        with open(input_file) as input_file:
            fold_results = json.load(input_file)

        for key_i, value in fold_results.items():
            if isinstance(value, dict):
                if key_i == type:
                    __precision += value['precision']
                    __recall += value['recall']
                    __fscore  += value['fscore']


        return self.filter_nan(__precision), self.filter_nan(__recall), self.filter_nan(__fscore)
    
    
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
            
            
    def report_IR_metrics(self, pattern, source_type='overall', verbose=False, has_filters=False):
        bert_file = pattern + '_base.json'
        bert_file_A = pattern + '_fe.json' # for frame-elements filter
        bert_file_B = pattern + '_fa.json' # for frame-association filters   


        # <-------------------------------------------------------------------------------------- BERT    
        bert_precision, bert_recall, bert_f1score = self.get_ir_metrics(bert_file, type=source_type)

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
            
        if has_filters:

            # <-------------------------------------------------------------------------------------- BERT    
            bert_precision, bert_recall, bert_f1score = self.get_ir_metrics(bert_file_A, type=source_type)

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
            bert_precision, bert_recall, bert_f1score = self.get_ir_metrics(bert_file_B, type=source_type)

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
                

    def compare_BERT_to_answer_bot(self, ansbot, pattern, source_type='so', verbose=False, has_filters=False, aplha=0.1):
        bert_file = pattern + '_base.json'
        bert_file_A = pattern + '_fe.json' # for frame-elements filter
        bert_file_B = pattern + '_fa.json' # for frame-association filters   
        
        
        answerbot_precision, answerbot_recall, _ = self.get_ir_metrics_to_list(ansbot, type='so')
        


        # <-------------------------------------------------------------------------------------- BERT
        _id = bert_file.replace("output/", "").replace("_base.json", "")
        _precision, _recall, _f1score = self.get_fold_metrics_to_list(bert_file, type=source_type)
        
        
        _r = scipy.stats.mannwhitneyu(answerbot_precision, _precision, alternative='two-sided')
        if _r.pvalue <= aplha:
            effect_size = self.cohend(answerbot_precision, _precision)
            logger.info("p-value=" +Fore.RED +  "{:.3f} ".format(_r.pvalue) + Style.RESET_ALL +\
                        "effect-size=" +Fore.RED +  "{:.3f} ".format(effect_size) + Style.RESET_ALL +\
                        " for " + Fore.RED +  f"{_id}" + Style.RESET_ALL + " [PRECISION]")
            
        _r = scipy.stats.mannwhitneyu(answerbot_recall, _recall, alternative='two-sided')
        if _r.pvalue <= aplha:
            effect_size = self.cohend(answerbot_recall, _recall)
            logger.info("p-value=" +Fore.RED +  "{:.3f} ".format(_r.pvalue) + Style.RESET_ALL +\
                        "effect-size=" +Fore.RED +  "{:.3f} ".format(effect_size) + Style.RESET_ALL +\
                        " for " + Fore.RED +  f"{_id}" + Style.RESET_ALL + " [RECALL]")
            
            
        # <-------------------------------------------------------------------------------------- BERT  
        logger.info("")
        _id = bert_file_A.replace("output/", "").replace("_fe.json", " w/ frame-elements")            
        _precision, _recall, _f1score = self.get_fold_metrics_to_list(bert_file_A, type=source_type)
        
        
        _r = scipy.stats.mannwhitneyu(answerbot_precision, _precision, alternative='two-sided')
        if _r.pvalue <= aplha:
            effect_size = self.cohend(answerbot_precision, _precision)
            logger.info("p-value=" +Fore.RED +  "{:.3f} ".format(_r.pvalue) + Style.RESET_ALL +\
                        "effect-size=" +Fore.RED +  "{:.3f} ".format(effect_size) + Style.RESET_ALL +\
                        " for " + Fore.RED +  f"{_id}" + Style.RESET_ALL + " [PRECISION]")
            
        _r = scipy.stats.mannwhitneyu(answerbot_recall, _recall, alternative='two-sided')
        if _r.pvalue <= aplha:
            effect_size = self.cohend(answerbot_recall, _recall)
            logger.info("p-value=" +Fore.RED +  "{:.3f} ".format(_r.pvalue) + Style.RESET_ALL +\
                        "effect-size=" +Fore.RED +  "{:.3f} ".format(effect_size) + Style.RESET_ALL +\
                        " for " + Fore.RED +  f"{_id}" + Style.RESET_ALL + " [RECALL]")
            

        # <-------------------------------------------------------------------------------------- BERT            
        logger.info("")
        _id = bert_file_B.replace("output/", "").replace("_fa.json", " w/ frame-associations")
        _precision, _recall, _f1score = self.get_fold_metrics_to_list(bert_file_B, type=source_type)
        
        
        _r = scipy.stats.mannwhitneyu(answerbot_precision, _precision, alternative='two-sided')
        if _r.pvalue <= aplha:
            effect_size = self.cohend(answerbot_precision, _precision)
            logger.info("p-value=" +Fore.RED +  "{:.3f} ".format(_r.pvalue) + Style.RESET_ALL +\
                        "effect-size=" +Fore.RED +  "{:.3f} ".format(effect_size) + Style.RESET_ALL +\
                        " for " + Fore.RED +  f"{_id}" + Style.RESET_ALL + " [PRECISION]")
            
        _r = scipy.stats.mannwhitneyu(answerbot_recall, _recall, alternative='two-sided')
        if _r.pvalue <= aplha:
            effect_size = self.cohend(answerbot_recall, _recall)
            logger.info("p-value=" +Fore.RED +  "{:.3f} ".format(_r.pvalue) + Style.RESET_ALL +\
                        "effect-size=" +Fore.RED +  "{:.3f} ".format(effect_size) + Style.RESET_ALL +\
                        " for " + Fore.RED +  f"{_id}" + Style.RESET_ALL + " [RECALL]")
            
            

    def compare_IR_to_answer_bot(self, ansbot, pattern, source_type='so', verbose=False, has_filters=False, aplha=0.1):
        bert_file = pattern + '_base.json'
        bert_file_A = pattern + '_fe.json' # for frame-elements filter
        bert_file_B = pattern + '_fa.json' # for frame-association filters   
        
        
        answerbot_precision, answerbot_recall, _ = self.get_ir_metrics_to_list(ansbot, type='so')
        


        # <-------------------------------------------------------------------------------------- BERT
        _id = bert_file.replace("output/", "").replace("_base.json", "")
        _precision, _recall, _f1score = self.get_ir_metrics_to_list(bert_file, type=source_type)
        
        
        _r = scipy.stats.mannwhitneyu(answerbot_precision, _precision, alternative='two-sided')
        if _r.pvalue <= aplha:
            effect_size = self.cohend(answerbot_precision, _precision)
            logger.info("p-value=" +Fore.RED +  "{:.3f} ".format(_r.pvalue) + Style.RESET_ALL +\
                        "effect-size=" +Fore.RED +  "{:.3f} ".format(effect_size) + Style.RESET_ALL +\
                        " for " + Fore.RED +  f"{_id}" + Style.RESET_ALL + " [PRECISION]")
            
        _r = scipy.stats.mannwhitneyu(answerbot_recall, _recall, alternative='two-sided')
        if _r.pvalue <= aplha:
            effect_size = self.cohend(answerbot_recall, _recall)
            logger.info("p-value=" +Fore.RED +  "{:.3f} ".format(_r.pvalue) + Style.RESET_ALL +\
                        "effect-size=" +Fore.RED +  "{:.3f} ".format(effect_size) + Style.RESET_ALL +\
                        " for " + Fore.RED +  f"{_id}" + Style.RESET_ALL + " [RECALL]")
            
            
        # <-------------------------------------------------------------------------------------- BERT  
        logger.info("")
        _id = bert_file_A.replace("output/", "").replace("_fe.json", " w/ frame-elements")            
        _precision, _recall, _f1score = self.get_ir_metrics_to_list(bert_file_A, type=source_type)
        
        
        _r = scipy.stats.mannwhitneyu(answerbot_precision, _precision, alternative='two-sided')
        if _r.pvalue <= aplha:
            effect_size = self.cohend(answerbot_precision, _precision)
            logger.info("p-value=" +Fore.RED +  "{:.3f} ".format(_r.pvalue) + Style.RESET_ALL +\
                        "effect-size=" +Fore.RED +  "{:.3f} ".format(effect_size) + Style.RESET_ALL +\
                        " for " + Fore.RED +  f"{_id}" + Style.RESET_ALL + " [PRECISION]")
            
        _r = scipy.stats.mannwhitneyu(answerbot_recall, _recall, alternative='two-sided')
        if _r.pvalue <= aplha:
            effect_size = self.cohend(answerbot_recall, _recall)
            logger.info("p-value=" +Fore.RED +  "{:.3f} ".format(_r.pvalue) + Style.RESET_ALL +\
                        "effect-size=" +Fore.RED +  "{:.3f} ".format(effect_size) + Style.RESET_ALL +\
                        " for " + Fore.RED +  f"{_id}" + Style.RESET_ALL + " [RECALL]")
            

        # <-------------------------------------------------------------------------------------- BERT            
        logger.info("")
        _id = bert_file_B.replace("output/", "").replace("_fa.json", " w/ frame-associations")
        _precision, _recall, _f1score = self.get_ir_metrics_to_list(bert_file_B, type=source_type)
        
        
        _r = scipy.stats.mannwhitneyu(answerbot_precision, _precision, alternative='two-sided')
        if _r.pvalue <= aplha:
            effect_size = self.cohend(answerbot_precision, _precision)
            logger.info("p-value=" +Fore.RED +  "{:.3f} ".format(_r.pvalue) + Style.RESET_ALL +\
                        "effect-size=" +Fore.RED +  "{:.3f} ".format(effect_size) + Style.RESET_ALL +\
                        " for " + Fore.RED +  f"{_id}" + Style.RESET_ALL + " [PRECISION]")
            
        _r = scipy.stats.mannwhitneyu(answerbot_recall, _recall, alternative='two-sided')
        if _r.pvalue <= aplha:
            effect_size = self.cohend(answerbot_recall, _recall)
            logger.info("p-value=" +Fore.RED +  "{:.3f} ".format(_r.pvalue) + Style.RESET_ALL +\
                        "effect-size=" +Fore.RED +  "{:.3f} ".format(effect_size) + Style.RESET_ALL +\
                        " for " + Fore.RED +  f"{_id}" + Style.RESET_ALL + " [RECALL]")            
            
        

            
    def cohend(self, d1, d2):
        # calculate the size of samples
        n1, n2 = len(d1), len(d2)
        # calculate the variance of the samples
        s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
        # calculate the pooled standard deviation
        s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        # calculate the means of the samples
        u1, u2 = mean(d1), mean(d2)
        # calculate the effect size
        return (u1 - u2) / s
