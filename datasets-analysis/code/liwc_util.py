# -*- coding: utf-8 -*-
import collections
import regex as re

def load_liwc(filename):
    """
    Load LIWC dataset
    input: a file that stores LIWC 2007 English dataset
    output:
        result: a dictionary that maps each word to the LIWC cluster ids
            that it belongs to
        class_id: a dict that maps LIWC cluster to category id,
            this does not seem useful, here for legacy reasons
        cluster_result: a dict that maps LIWC cluster id to all words
            in that cluster
        categories: a dict that maps LIWC cluster to its name
        category_reverse: a dict that maps LIWC cluster name to its id
    """
    # be careful, there can be *s
    result = collections.defaultdict(set)
    cluster_result = collections.defaultdict(set)
    class_id, cid = {}, 1
    categories, prefixes = {}, set()
    number = re.compile('\d+')
    with open(filename) as fin:
        start_cat, start_word = False, False
        for line in fin:
            line = line.strip()
            if start_cat and line == '%':
                start_word = True
                continue
            if line == '%':
                start_cat = True
                continue
            if start_cat and not start_word:
                parts = line.split()
                categories[int(parts[0])] = parts[1]
                continue
            if not start_word:
                continue
            parts = line.split()
            w = parts[0]
            if w.endswith('*'):
                prefixes.add(w)
            for c in parts[1:]:
                cs = re.findall(number, c)
                for n in cs:
                    n = int(n)
                    cluster_result[n].add(w)
                    result[w].add(n)
                    if n not in class_id:
                        class_id[n] = cid
                        cid += 1
    category_reverse = {v: k for (k ,v) in categories.items()}
    return result, class_id, cluster_result, categories, category_reverse


def in_liwc_cluster(w, cluster):
    if w in cluster:
        return True
    for i in range(1, len(w)):
        k = '%s*' % w[:i]
        if k in cluster:
            return True
    return False


def count_liwc_words(words, cluster=None):
    return sum([in_liwc_cluster(w, cluster) for w in words])

