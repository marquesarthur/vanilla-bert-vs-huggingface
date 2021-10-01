import logging
import os
from abc import ABC, abstractmethod
from collections import Counter, defaultdict


import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import Adam
from tensorflow.python import keras
from tqdm import tqdm
from transformers import (AutoTokenizer, BertTokenizerFast, DistilBertConfig,
                          DistilBertTokenizerFast,
                          TFBertForSequenceClassification,
                          TFDistilBertForSequenceClassification,
                          TFDistilBertModel)

from data import add_raw_data, get_class_weights, get_ds_synthetic_data, undersample_df
from metrics import MetricsAggregator
from seframe_filters import AssociationPairFilter, FrameElementFilter

USE_TPU = False
os.environ['TF_KERAS'] = '1'


logger = logging.getLogger()


class TFBertForTaskTextClassification(ABC):

    def __init__(self, model_id='bert-base-uncased') -> None:
        super().__init__()
        self.model_id = model_id

        # Bert Model Constants
        self.seq_len = 64
        self.batch_size = 64  # 64 32 larger batch size causes OOM errors
        self.epochs = 10
        self.lr = 1e-5

        # Various script configuration constants
        self.use_frame_filtering = False
        self.match_frame_from_task = False
        self.sentence_task_frame_pairs = None

        self.use_pyramid = False

        self.undersampling = True
        # ratio of how many samples from 0-class, to 1-class, e.g.: 2:1
        self.n_undersampling = 2
        self.use_ds_synthetic = False
        self.min_w = 3  # for the DS-synthetic dataset, this sets the minimum number of annotators who marked that sentence

        self.fn_frame_elements = FrameElementFilter()
        self.fn_frame_pairs = AssociationPairFilter()

        self.target_output = 10
        self.metrics = MetricsAggregator()

    def tokenizer(self, cache_dir='/home/msarthur/scratch', local_files_only=True):
        if self.model_id == 'distilbert-base-uncased':
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(
                self.model_id, cache_dir=cache_dir, local_files_only=local_files_only)
        else:
            self.tokenizer = BertTokenizerFast.from_pretrained(
                self.model_id, cache_dir=cache_dir, local_files_only=local_files_only)

    @abstractmethod
    def get_train_val_test(self, corpus, task_uid, size=0.9):
        pass

    def test(self, source, df_test, model):
        df_source = df_test[df_test["source"] == source]
        task_title = df_source['question'].tolist()[0]
        text = df_source['text'].tolist()
        pweights = df_source['weights'].tolist()

        # Encode X_test
        test_encodings = self.encode(df_source)
        test_labels = df_source['category_index'].tolist()

        test_dataset = tf.data.Dataset.from_tensor_slices((
            dict(test_encodings),
            test_labels
        ))

        y_true = [y.numpy() for x, y in test_dataset]

        # means that this source has at least one annotated sentence
        if any([k == 1 for k in y_true]):
            y_predict, y_probs = self.eval_model(model, test_dataset)

            if self.match_frame_from_task:
                y_true, y_predict = self.apply_association_filter(
                    task_title, text, y_true, y_predict, y_probs, self.sentence_task_frame_pairs)

            if self.use_frame_filtering:
                y_true, y_predict = self.apply_frame_element_filter(
                    task_title, text, y_true, y_predict, y_probs)

            if len(y_true) > 0 and len(y_predict) > 0:
                accuracy = accuracy_score(y_true, y_predict)
                macro_f1 = f1_score(y_true, y_predict, average='macro')

                self.metrics.classification_report_lst.append(
                    classification_report(y_true, y_predict))
                self.metrics.aggregate_report_metrics(
                    classification_report(y_true, y_predict, output_dict=True))

                self.metrics.log_results(y_true, y_predict, accuracy, macro_f1)

                precision, recall, fscore, _ = precision_recall_fscore_support(
                    y_true, y_predict, average='macro')

                self.metrics.aggregate_macro_metrics(
                    self.metrics.prediction_metrics, precision, recall, fscore)
                self.metrics.aggregate_macro_source_metrics(
                    precision, recall, fscore, source)

                logger.info("Precision: {:.4f}".format(precision))
                logger.info("Recall: {:.4f}".format(recall))
                logger.info("F1: {:.4f}".format(fscore))

                self.metrics.log_examples(
                    task_title, source, text, pweights, y_predict, y_probs, k=self.target_output)
                self.metrics.log_venn_diagram(y_true, y_predict, text)
                self.metrics.source_lst.append(source)

    def eval_model(self, model, test_data):
        preds = model.predict(test_data.batch(1)).logits

        # transform to array with probabilities
        res = tf.nn.softmax(preds, axis=1).numpy()

        y_predict, y_probs = res.argmax(axis=-1), res[:, 1]
        aux = [(idx, prob) for idx, prob in enumerate(y_probs)]

        max_pred_values = self.target_output

        cnt = 0
        for idx, prob in sorted(aux, key=lambda k: k[1], reverse=True):
            cnt += 1
            if cnt > max_pred_values:
                y_predict[idx] = 0

        return y_predict, y_probs

    def build(self, train_dataset, val_dataset, weights, checkpoint_filepath='/home/msarthur/scratch/best_model', cache_dir='/home/msarthur/scratch', local_files_only=True):
        if self.model_id == 'distilbert-base-uncased':
            model = TFDistilBertForSequenceClassification.from_pretrained(
                self.model_id, cache_dir=cache_dir, local_files_only=local_files_only
            )
        else:
            model = TFBertForSequenceClassification.from_pretrained(
                self.model_id, cache_dir=cache_dir, local_files_only=local_files_only
            )

        # freeze all the parameters
        # for param in model.parameters():
        #   param.requires_grad = False

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)

        METRICS = [
            tf.keras.metrics.SparseCategoricalAccuracy()
        ]

        early_stopper = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', mode='min', patience=4,
            verbose=1, restore_best_weights=True
        )

        # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint

        mc = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_filepath,
            monitor='val_loss', mode='min', verbose=1,
            save_best_only=True,
            save_weights_only=True
        )

        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=METRICS
        )

        # https://discuss.huggingface.co/t/how-to-dealing-with-data-imbalance/393/3
        # https://wandb.ai/ayush-thakur/huggingface/reports/Early-Stopping-in-HuggingFace-Examples--Vmlldzo0MzE2MTM
        model.fit(
            train_dataset.shuffle(1000).batch(self.batch_size),
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=weights,
            validation_data=val_dataset.shuffle(1000).batch(self.batch_size),
            callbacks=[early_stopper, mc]
        )

        model.load_weights(checkpoint_filepath)

        return model

    def encode(self, dataframe):

        seq_a = dataframe['text'].tolist()
        seq_b = dataframe['question'].tolist()

        return self.tokenizer(seq_a, seq_b, truncation=True, padding=True, max_length=self.seq_len)

    def __to_one_hot_encoding(self, data, nb_classes=2):
        targets = np.array([data]).reshape(-1)
        one_hot_targets = np.eye(nb_classes)[targets]
        return one_hot_targets

    def apply_frame_element_filter(self, task_title, text, y_true, y_predict, y_probs, relevant_class=1):
        y_true_prime = []
        y_predict_prime = []

        # update probs after k = 10, same as in eval_model
        aux = [(idx, prob) for idx, prob in enumerate(y_probs)]

        cnt = 0
        for idx, prob in sorted(aux, key=lambda k: k[1], reverse=True):
            y_true_prime.append(y_true[idx])
            _t = text[idx]

            cnt += 1
            if cnt > self.target_output:
                y_predict_prime.append(y_predict[idx])
            else:
                if self.fn_frame_elements.has_meaningful_frame(_t):
                    y_predict_prime.append(max(y_predict[idx], relevant_class))
                else:
                    y_predict_prime.append(y_predict[idx])

        return y_true_prime, y_predict_prime

    def apply_association_filter(self, task_title, text, y_true, y_predict, y_probs, task_filter, relevant_class=1):
        y_true_prime = []
        y_predict_prime = []

        # update probs after k = 10, same as in eval_model
        aux = [(idx, prob) for idx, prob in enumerate(y_probs)]
        max_pred_values = max(int(len(text) * 0.15), self.target_output)

        cnt = 0
        for idx, prob in sorted(aux, key=lambda k: k[1], reverse=True):
            y_true_prime.append(y_true[idx])
            _t = text[idx]

            cnt += 1
            if cnt > max_pred_values:
                y_predict_prime.append(y_predict[idx])
            else:
                if self.fn_frame_pairs.has_common_task_frame(task_title, _t, task_filter):
                    y_predict_prime.append(max(y_predict[idx], relevant_class))
                else:
                    y_predict_prime.append(y_predict[idx])

        return y_true_prime, y_predict_prime

    def get_evaluation_metrics(self):
        return self.metrics.prediction_metrics, \
            self.metrics.api_metrics, \
            self.metrics.so_metrics, \
            self.metrics.git_metrics, \
            self.metrics.misc_metrics


class TFBertForAndroidTaskTextClassification(TFBertForTaskTextClassification):

    def get_train_val_test(self, corpus, task_uid, size=0.9):
        if not isinstance(task_uid, list):
            task_uid = [task_uid]

        train_data_raw = defaultdict(list)
        test_data_raw = defaultdict(list)

        for _data in tqdm(corpus):
            if _data['question'] in task_uid:
                add_raw_data(test_data_raw, _data,
                             use_pyramid=self.use_pyramid)
            else:
                add_raw_data(train_data_raw, _data,
                             use_pyramid=self.use_pyramid)

        train_val = pd.DataFrame.from_dict(train_data_raw)
        test = pd.DataFrame.from_dict(test_data_raw)

        # https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
        #  randomize rows....
        train_val = train_val.sample(frac=1).reset_index(drop=True)
        test = test.sample(frac=1).reset_index(drop=True)

        if self.undersampling:
            train_val = undersample_df(train_val, n_times=self.n_undersampling)
            train_val = train_val.sample(frac=1).reset_index(drop=True)

        weights = get_class_weights(train_val['category_index'].tolist())

        # split data for training and validation. stratifies splitting based on y labels
        train, val = train_test_split(
            train_val,
            stratify=train_val['category_index'].tolist(),
            train_size=size
        )

        return train, val, test, weights


class TFBertForSyntheticTaskTextClassification(TFBertForTaskTextClassification):

    def get_train_val_test(self, corpus, task_uid, size=0.9):
        if not isinstance(task_uid, list):
            task_uid = [task_uid]

        train_data_raw = defaultdict(list)
        test_data_raw = defaultdict(list)

        for _data in tqdm(corpus):
            if _data['question'] in task_uid:
                add_raw_data(test_data_raw, _data, use_pyramid=self.use_pyramid)
        
    
        train_val = get_ds_synthetic_data(undersample_n=self.n_undersampling, min_w=self.min_w)
        test = pd.DataFrame.from_dict(test_data_raw)

        
        # https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
        #  randomize rows....
        train_val = train_val.sample(frac=1).reset_index(drop=True)
        test = test.sample(frac=1).reset_index(drop=True)

        weights = get_class_weights(train_val['category_index'].tolist())

        # split data for training and validation. stratifies splitting based on y labels
        train, val = train_test_split(
            train_val,
            stratify=train_val['category_index'].tolist(),
            train_size=size
        )

        return train, val, test, weights    
    
