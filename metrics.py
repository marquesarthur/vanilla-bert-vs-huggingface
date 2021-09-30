
import logging
from collections import defaultdict

import numpy as np
from colorama import Fore, Style

logger = logging.getLogger()


class MetricsAggregator(object):

    def __init__(self) -> None:
        super().__init__()
        # varios lists that will keep the metrics gathered by the model

        self.clz_report_lst = defaultdict(list)
        self.recommendation_metrics = defaultdict(list)
        self.prediction_metrics = defaultdict(list)
        self.api_metrics = defaultdict(list)
        self.so_metrics = defaultdict(list)
        self.git_metrics = defaultdict(list)
        self.misc_metrics = defaultdict(list)

        self.classification_report_lst = []
        self.log_examples_lst = []
        self.source_lst = []
        self.venn_diagram_set = []

    def reset_aggregators(self) -> None:
        self.clz_report_lst = defaultdict(list)
        self.recommendation_metrics = defaultdict(list)
        self.prediction_metrics = defaultdict(list)
        self.api_metrics = defaultdict(list)
        self.so_metrics = defaultdict(list)
        self.git_metrics = defaultdict(list)
        self.misc_metrics = defaultdict(list)

    def aggregate_macro_metrics(self, store_at, precision, recall, fscore) -> None:
        store_at['precision'].append(precision)
        store_at['recall'].append(recall)
        store_at['fscore'].append(fscore)

    def aggregate_macro_source_metrics(self, precision, recall, fscore, source) -> None:
        s = source
        if 'docs.oracle' in s or 'developer.android' in s:
            self.aggregate_macro_metrics(
                self.api_metrics, precision, recall, fscore)
        elif 'stackoverflow.com' in s:
            self.aggregate_macro_metrics(
                self.so_metrics, precision, recall, fscore)
        elif 'github.com' in s:
            self.aggregate_macro_metrics(
                self.git_metrics, precision, recall, fscore)
        elif 'github.com' not in s and 'docs.oracle' not in s and 'developer.android' not in s and 'stackoverflow.com' not in s:
            self.aggregate_macro_metrics(
                self.misc_metrics, precision, recall, fscore)

    def aggregate_recommendation_metrics(self, store_at, k, precision_at_k, pyramid_precision_at_k) -> None:
        store_at['k'].append(k)
        store_at['precision'].append(precision_at_k)
        store_at['∆ precision'].append(pyramid_precision_at_k)

    def aggregate_report_metrics(self, clz_report) -> None:
        relevant_label = str(1)
        if relevant_label in clz_report:
            for _key in ['precision', 'recall']:
                if _key in clz_report[relevant_label]:
                    self.clz_report_lst[_key].append(
                        clz_report[relevant_label][_key])

    def log_examples(self, task_title, source, text, pweights, y_predict, y_probs, k=10) -> None:
        # get the predicted prob at every index
        idx_probs = [(idx, y_predict[idx], y_probs[idx])
                     for idx, _ in enumerate(y_predict)]

        # filter probs for all indexes predicted as relevant
        idx_probs = list(filter(lambda k: k[1] == 1, idx_probs))

        most_probable = sorted(idx_probs, key=lambda i: i[2], reverse=True)

        result = [idx for idx, _, _ in most_probable][:k]

        for idx in result:
            self.log_examples_lst.append((
                source,
                task_title,
                pweights[idx],
                y_predict[idx],
                y_probs[idx],
                text[idx]
            ))

    def log_venn_diagram(self, y_true, y_predicted, text) -> None:
        cnt = 0
        try:
            for _true, _predict, _t in zip(y_true, y_predicted, text):
                if _true == 1 and _predict == 1:
                    cnt += 1
                    self.venn_diagram_set.append(_t)
        except Exception as ex:
            logger.info(str(ex))
        logger.info(Fore.RED + str(cnt) + Style.RESET_ALL + " entries logged")

    @staticmethod
    def avg_macro_metric_for(data):
        __precision = data['precision']
        __recall = data['recall']
        __fscore = data['fscore']

        return np.mean(__precision), np.mean(__recall), np.mean(__fscore)

    @staticmethod
    def log_results(y_true, y_predict, accuracy, macro_f1) -> None:
        logger.info("-" * 20)

        logger.info("Y")
        logger.info("[0s] {} [1s] {}".format(
            len(list(filter(lambda k: k == 0, y_true))),
            len(list(filter(lambda k: k == 1, y_true)))
        ))

        logger.info("predicted")
        logger.info("[0s] {} [1s] {}".format(
            len(list(filter(lambda k: k == 0, y_predict))),
            len(list(filter(lambda k: k == 1, y_predict)))
        ))

        logger.info("-" * 20)

        logger.info("Accuracy: {:.4f}".format(accuracy))
        logger.info("macro_f1: {:.4f}".format(macro_f1))

    @staticmethod
    def add_idx_fold_results(idx_split, store_at, prediction_metrics, api_metrics, so_metrics, git_metrics, misc_metrics) -> None:
        if idx_split not in store_at:
            store_at[idx_split] = dict()
            store_at[idx_split]['run_cnt'] = 0
            store_at[idx_split]['overall'] = defaultdict(list)
            store_at[idx_split]['api'] = defaultdict(list)
            store_at[idx_split]['so'] = defaultdict(list)
            store_at[idx_split]['git'] = defaultdict(list)
            store_at[idx_split]['misc'] = defaultdict(list)

        store_at[idx_split]['run_cnt'] += 1

        _precision, _recall, _f1score = MetricsAggregator.avg_macro_metric_for(
            prediction_metrics)
        store_at[idx_split]['overall']['precision'].append(_precision)
        store_at[idx_split]['overall']['recall'].append(_recall)
        store_at[idx_split]['overall']['fscore'].append(_f1score)

        _precision, _recall, _f1score = MetricsAggregator.avg_macro_metric_for(
            api_metrics)
        store_at[idx_split]['api']['precision'].append(_precision)
        store_at[idx_split]['api']['recall'].append(_recall)
        store_at[idx_split]['api']['fscore'].append(_f1score)

        _precision, _recall, _f1score = MetricsAggregator.avg_macro_metric_for(
            so_metrics)
        store_at[idx_split]['so']['precision'].append(_precision)
        store_at[idx_split]['so']['recall'].append(_recall)
        store_at[idx_split]['so']['fscore'].append(_f1score)

        _precision, _recall, _f1score = MetricsAggregator.avg_macro_metric_for(
            git_metrics)
        store_at[idx_split]['git']['precision'].append(_precision)
        store_at[idx_split]['git']['recall'].append(_recall)
        store_at[idx_split]['git']['fscore'].append(_f1score)

        _precision, _recall, _f1score = MetricsAggregator.avg_macro_metric_for(
            misc_metrics)
        store_at[idx_split]['misc']['precision'].append(_precision)
        store_at[idx_split]['misc']['recall'].append(_recall)
        store_at[idx_split]['misc']['fscore'].append(_f1score)

    @staticmethod
    def get_full_exec_results(fold_results):
        __precision, __recall, __fscore = [], [], []

        for key_i, value in fold_results.items():
            if isinstance(value, dict):
                for key_j, __data in value.items():
                    if key_j == 'overall':
                        logger.info(Fore.YELLOW + f"{key_i}" + Style.RESET_ALL)
                        logger.info("precision: " + Fore.RED +
                                    "{:.3f}".format(np.mean(__data['precision'])) + Style.RESET_ALL +
                                    f" {str([round(x, 2) for x in __data['precision']])}")
                        logger.info("recall:    " + Fore.RED +
                                    "{:.3f}".format(np.mean(__data['recall'])) + Style.RESET_ALL +
                                    f" {str([round(x, 2) for x in __data['recall']])}")
                        logger.info("f1-score:  " +
                                    Fore.RED + "{:.3f}".format(np.mean(__data['fscore'])) + Style.RESET_ALL +
                                    f" {str([round(x, 2) for x in __data['fscore']])}")

                        __precision += __data['precision']
                        __recall += __data['recall']
                        __fscore += __data['fscore']

        __precision = [x for x in __precision if str(x) != 'nan']
        __recall = [x for x in __recall if str(x) != 'nan']
        __fscore = [x for x in __fscore if str(x) != 'nan']

        return __precision, __recall, __fscore

    def examples_per_source_type(self, source_type='misc', n_samples=None) -> None:

        _sources = list(set([x[0] for x in self.log_examples_lst]))

        _template = "[w={}]" + Fore.RED + "[y={}]" + \
            Fore.YELLOW + "[p={:.4f}]" + Style.RESET_ALL + " {}"

        idx = 0
        for s in _sources:
            examples_in_source = []
            if source_type == 'api' and ('docs.oracle' in s or 'developer.android' in s):
                examples_in_source = list(
                    filter(lambda k: k[0] == s, self.log_examples_lst))
                task_title = examples_in_source[0][1]
                idx += 1
            elif source_type == 'so' and ('stackoverflow.com' in s):
                examples_in_source = list(
                    filter(lambda k: k[0] == s, self.log_examples_lst))
                task_title = examples_in_source[0][1]
                idx += 1
            elif source_type == 'git' and ('github.com' in s):
                examples_in_source = list(
                    filter(lambda k: k[0] == s, self.log_examples_lst))
                task_title = examples_in_source[0][1]
                idx += 1
            elif source_type == 'misc' and 'github.com' not in s and 'docs.oracle' not in s and 'developer.android' not in s and 'stackoverflow.com' not in s:
                examples_in_source = list(
                    filter(lambda k: k[0] == s, self.log_examples_lst))
                task_title = examples_in_source[0][1]
                idx += 1
            if not examples_in_source:
                continue
            logger.info('')
            logger.info(Fore.RED + f"{task_title}" + Style.RESET_ALL)
            logger.info(s)
            logger.info('')

            for _, _, pweights, y_predict, y_probs, text in examples_in_source:
                logger.info(_template.format(
                    pweights, y_predict, y_probs, text))
                logger.info('')
            logger.info('-' * 20)

            if n_samples and idx >= n_samples:
                break
