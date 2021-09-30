import json
from collections import Counter, defaultdict
from itertools import combinations, product


class FrameElementFilter(object):

    def __init__(self) -> None:
        super().__init__()
        seframes = {}
        with open('seframes.json') as input_file:
            self.seframes = json.load(input_file)

    def has_meaningful_frame(self, text):
        meaning_frames = [
            'Using', 'Being_obligated', 'Required_event', 'Causation', 'Attempt', 'Execution'
        ]

        if text in self.seframes:
            text_labels = self.seframes[text]
            if any([elem in meaning_frames for elem in text_labels]):
                return True

        return False


class AssociationPairFilter(object):

    def __init__(self) -> None:
        super().__init__()
        seframes = {}
        with open('seframes.json') as input_file:
            self.seframes = json.load(input_file)

    def get_most_common_frame_relationships(self, df_train, n_of_rules=100):
        frame_task_pairs = []
        df_filtered = df_train[df_train['category_index'] == 1]
        for __task, __text in zip(df_filtered['question'].tolist(), df_filtered['text'].tolist()):

            task_labels, text_labels = [], []
            if __task in self.seframes:
                task_labels = self.seframes[__task]

            if __text in self.seframes:
                text_labels = self.seframes[__text]

            if task_labels and text_labels:
                all_pairs = list(product(task_labels, text_labels))
                frame_task_pairs += all_pairs

        most_common_frame_relationships = [pair for pair, cnt in Counter(
            frame_task_pairs).most_common(n_of_rules)]
        return most_common_frame_relationships

    def has_common_task_frame(self, task_title, text, most_common_frame_relationships):
        task_labels, text_labels = [], []
        if task_title in self.seframes:
            task_labels = self.seframes[task_title]
        else:
            return False

        if text in self.seframes:
            text_labels = self.seframes[text]
        else:
            return False

        all_pairs = list(product(task_labels, text_labels))
        has_frame_match = any(
            [elem in most_common_frame_relationships for elem in all_pairs])

        return has_frame_match
