import itertools
import json
import logging
import os
import sys
from pathlib import Path

from Levenshtein import ratio
from colorama import Fore, Style

logger = logging.getLogger()
logger.level = logging.DEBUG


def get_input_for_BERT():
    result = []

    data_batches = ["batch1", "batch2", "batch3", "batch4", "batch5"]
    for batch in data_batches:
        all_annotations, api_annotations, git_annotations, so_annotations, misc_annotations = {}, {}, {}, {}, {}
        input_folder = os.path.join(
            "expert_tasks", "task_batches", batch
        )

        answers_folder = os.path.join(
            "expert_answers", batch
        )

        # Side effect of this method is to update each dictionary with the text marked by annotators
        process_batch(answers_folder, all_annotations, api_annotations, git_annotations, so_annotations, misc_annotations)

        result += _process_input(input_folder, api_annotations, git_annotations, so_annotations, misc_annotations)

    final_data = _flatten(result)
    return final_data


def _flatten(source_data):
    result = []

    for task_title, task_full_description, source, text_in_source, relevant_sentences_in_source in source_data:

        for __text in text_in_source:
            value = 0
            if __text in relevant_sentences_in_source:
                value = relevant_sentences_in_source[__text]

            result.append(dict(
                text=__text,
                question=task_title,
                source=source,
                category_index= 1 if value > 0 else 0,
                weights=value
            ))

    return result





def _process_input(input_folder, api_annotations, git_annotations, so_annotations, misc_annotations):
    encoding = "UTF-8" if os.name != "nt" else "mbcs"
    result = []
    try:
        for task_path in Path(input_folder).rglob('*.json'):
            with open(task_path, encoding=encoding) as fi:
                # with open(path) as fi:
                data = json.load(fi)
                task_title = data['task']
                task_full_description = data['description']  # array
                result += get_all_the_text_for(data['resources']['api'], task_title, task_full_description, api_annotations, git_annotations, so_annotations, misc_annotations, type='API')
                result += get_all_the_text_for(data['resources']['qa'], task_title, task_full_description, api_annotations, git_annotations, so_annotations, misc_annotations, type='SO')
                result += get_all_the_text_for(data['resources']['git'], task_title, task_full_description, api_annotations, git_annotations, so_annotations, misc_annotations, type='GIT')
                result += get_all_the_text_for(data['resources']['other'], task_title, task_full_description, api_annotations, git_annotations, so_annotations, misc_annotations,  type='MISC')
                # logger.info("done")

    except Exception as ex:
        logger.error(str(ex))

    return result

def get_all_the_text_for(resource, task_title, task_full_description, api_annotations, git_annotations, so_annotations, misc_annotations,  type='SO'):
    source_data = []
    for artifact in resource:
        page_title = artifact['title']
        uri = artifact['link']

        if type == 'SO':
            all_text = list(itertools.chain.from_iterable([q['text'] for q in artifact['answers']]))
        else:
            all_text = artifact['content']

        if type == 'API':
            all_relevant_text = get_relevant_text_based_on_annotations(page_title, all_text, api_annotations)
        elif type == 'SO':
            all_relevant_text = get_relevant_text_based_on_annotations(page_title, all_text, so_annotations)
        elif type == 'GIT':
            all_relevant_text = get_relevant_text_based_on_annotations(page_title, all_text, git_annotations)
        else:
            all_relevant_text = get_relevant_text_based_on_annotations(page_title, all_text, misc_annotations)

        if __information_density(all_text, all_relevant_text):
            logger.info(
                Fore.RED + f'{len(all_relevant_text)} ' + Fore.YELLOW + f'{len(all_text)} ' + Style.RESET_ALL + ' ' + uri)

            source_data.append((task_title, task_full_description, uri, all_text, all_relevant_text))

    return source_data


def __information_density(all_text, all_relevant_text, threshold=0.10, min_count=3):
    if not all_relevant_text:
        return False

    if not all_text:
        return False

    text_cnt = len(all_text)
    relevant_cnt = len(all_relevant_text)

    # return relevant_cnt / float(text_cnt) >= threshold and relevant_cnt >= min_count
    return relevant_cnt >= min_count


def get_relevant_text_based_on_annotations(page_title, all_text, annotations, min_w=1):
    result = {}
    text_marked = None
    for key in annotations.keys():
        if ratio(__strip_title(key), __strip_title(page_title)) >= 0.65:
            text_marked = annotations[key]
            break

    if text_marked:
        for text in all_text:
            for _t, _annotators in text_marked.items():
                if ratio(text, _t) >= 0.70:
                    result[text] = len(_annotators)
                    break

    return result

def __strip_title(data):
    return data.lower().replace('android developers', '').replace('stackoverflow', '').replace(
        'codepath android cliffnotes', '').replace('medium', '').replace('|', '').strip()




def process_batch(answers_folder, all_annotations, api_annotations, git_annotations, so_annotations,
                  misc_annotations):

    annotations = get_all_annotations(answers_folder, prefix="API")
    api_annotations.update(group_annotations(annotations))
    all_annotations.update(api_annotations)  # merge dictionaries

    annotations = get_all_annotations(answers_folder, prefix="GIT")
    git_annotations.update(group_annotations(annotations))
    all_annotations.update(git_annotations)  # merge dictionaries

    annotations = get_all_annotations(answers_folder, prefix="SO")
    so_annotations.update(group_annotations(annotations))
    all_annotations.update(so_annotations)  # merge dictionaries

    annotations = get_all_annotations(answers_folder, prefix="MISC")
    misc_annotations.update(group_annotations(annotations))
    all_annotations.update(misc_annotations)  # merge dictionaries


def get_all_annotations(answers_folder, prefix=""):
    SKIP_TAGS = ["PRE"]
    result = []
    encoding = "UTF-8" if os.name != "nt" else "mbcs"
    try:
        for path in Path(answers_folder).rglob(f'{prefix}*.json'):
            with open(path, encoding=encoding) as fi:
                # with open(path) as fi:
                _answer_data = json.load(fi)
                annotator_id = _answer_data['session_uid'] if 'session_uid' in _answer_data else 'unknown'
                if 'items' in _answer_data and _answer_data['items']:
                    title = next(iter(_answer_data['items']))['href']

                    raw_text = [item['text'] for item in _answer_data['items'] if item['tag'] not in SKIP_TAGS]
                    highlighted_text = raw_text

                    result += [dict(text=t, annotator=annotator_id, title=title) for t in highlighted_text]

    except Exception:
        logging.exception("message")

    return result

def group_annotations(raw_annotations):
    result = {}
    individual_pages_titles = list(set([r['title'] for r in raw_annotations]))

    for page in individual_pages_titles:

        all_highlighted_text = list(filter(lambda k: k['title'] == page, raw_annotations))
        all_highlighted_text = [d['text'] for d in all_highlighted_text]

        text_annotators = {}
        for text in all_highlighted_text:
            marked_by = list(filter(lambda k: k['text'] == text, raw_annotations))

            text_annotators[text] = list(set([m['annotator'] for m in marked_by]))

        result[page] = text_annotators

    return result