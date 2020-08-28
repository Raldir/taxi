#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import gensim
import csv
import io
import sys
import numpy as np
import gzip
import os
import argparse
import logging
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gensim.models.poincare import PoincareModel, PoincareRelations
from gensim.viz.poincare import poincare_2d_visualization
from gensim.test.utils import datapath
from data_loader import read_all_data, read_trial_data, read_input, compound_operator
import plotly.plotly as py
from nltk.corpus import wordnet as wn
#py.sign_in('RamiA', 'lAA8oTL51miiC79o3Hrz')

from collections import Counter


from sklearn import preprocessing
import numpy as np
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import KMeans
import pandas


def compare_to_gold(gold, taxonomy, model,  model_poincare = None, outliers = [], threshold_add = 0.4, new_nodes = [], log = "", write_file = ""):
    taxonomy_c = taxonomy.copy()
    global compound_operator
    removed_outliers = []
    for element in taxonomy_c:
        if (element[0].replace(' ', compound_operator), element[1].replace(' ', compound_operator)) in outliers:
            continue
        removed_outliers.append((element[0], element[1]))

    if new_nodes:
        for element in new_nodes:
            removed_outliers.append((element[0].replace(compound_operator, " "), element[1].replace(compound_operator, " ")))

    removed_outliers = list(set(removed_outliers))

    correct = 0
    for element in removed_outliers:
        for ele_g in gold:
            if element[0] == ele_g[0] and element[1] == ele_g[1]:
                correct+=1
                break
    precision = correct / float(len(removed_outliers))
    recall = correct / float(len(gold))
    print(str(recall).replace(".", ',') +'\t' + str(precision).replace(".", ',') + '\t' + str(2*precision *recall / (precision + recall)).replace(".", ',') + '\t' +  str(len(new_nodes)))
    if log != None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), log)
        with open(path + ".txt", 'w') as f:
            for element in outliers:
                f.write(element[0] + '\t' + element[1] + '\n')
            f.write("Elements Taxonomy:" + str(float(len(removed_outliers))))
            f.write(str((float(len(gold)))) + '\n')
            f.write("Correct: " + str(correct) + '\n')
            f.write("Precision: " + str(precision) + '\n')
            f.write("Recall: " + str(recall) + '\n')
            f.write("F1: " + str(2*precision *recall / (precision + recall)) + '\n')
            f.close()
    if write_file != None:
        path =  os.path.join(os.path.dirname(os.path.abspath(__file__)), write_file + ".csv")
        with open(path, 'w') as f:
            for i, element in enumerate(removed_outliers):
                f.write(str(i) + '\t' + str(element[0]) + '\t' + str(element[1])  + '\n')
        f.close()

    return removed_outliers


def get_parent(relations,child):
    for relation in relations:
        if child == relation[1]:
            return relation[2]
    return None

def get_rank(entity1, entity2, model, threshhold):
    rank_inv = None
    similarities_rev = model.wv.similar_by_word(entity1, threshhold)
    similarities_rev = [entry[0] for entry in similarities_rev]
    for j in range(len(similarities_rev)):
        temp_rev = similarities_rev[j]
        if entity2 == temp_rev:
            rank_inv = j
    return rank_inv

#do not need to check if words in vocab since outliers must be in vocab
#TODO could happen that outlier would connect to new outlier, but is not regarded, so currently adding all but outlier, so order of replacing outliers is not irrelevant
def connect_new_nodes(gold, taxonomy, model, model_poincare, threshold, no_parents, no_co, wordnet = False):
    structure = {}
    new_nodes = set([])
    new_relationships = []
    gold_nodes = [relation[0] for relation in gold] + [relation[1] for relation in gold]
    taxonomy_nodes = (set([relation[0] for relation in taxonomy] + [relation[1] for relation in taxonomy]))
    results_parents = []
    results_substring = []
    pairs_parents = []
    results_co = []
    pairs_co = []
    for element in gold_nodes:
        if element not in taxonomy_nodes:
            new_nodes.add(element)
    count = 0
    count_p = 0
    for node in new_nodes:
        if node.replace(" ", compound_operator) in model.wv:
            count+=1
    for node in new_nodes:
        if node.replace(" ", compound_operator) +'.n.01' in model_poincare.kv:
            count_p +=1

    # print(count, "in embedding")
    # print(count_p, "in poincare_embedding")

    relations = taxonomy.copy()
    for i in range(len(relations)):
        relations[i] = (relations[i][0].replace(" ", compound_operator), relations[i][1].replace(" ", compound_operator))

    for parent in [relation[1] for relation in relations]:
        structure[parent] = [relation[0] for relation in relations if relation[1] == parent]

    for node in new_nodes:
        node = node.replace(" ", compound_operator)
        result_co_min = 10000000
        pair_co_min  = 0
        result_parent_min = 10000000
        pair_parent_min = 0
        for key in structure:
            #print(key)
            if structure[key] == []:
                print("no children: " + key)
                continue
            cleaned_co_hyponyms = []
            if len(structure[key]) < 1:
                continue
            result_parent, pair_parent, result_co, pair_co  = get_rank(node, key, structure[key], model, model_poincare, no_parents, no_co, compound = True, wordnet = wordnet)
            if result_parent < result_parent_min and result_parent != 0:
                result_parent_min = result_parent
                pair_parent_min = pair_parent
            if result_co < result_co_min and result_co != 0:
                result_co_min = result_co
                pair_co_min = pair_co

        if result_parent_min != 10000000:
            results_parents.append(result_parent_min)
            pairs_parents.append(pair_parent_min)
        if result_co_min != 10000000:
            results_co.append(result_co_min)
            pairs_co.append(pair_co_min)
        elif node.split('_')[0] in structure:
            results_substring.append((node, node.split('_')[0]))
        elif node.split('_')[-1] in structure:
            results_substring.append((node, node.split('_')[-1]))


    results_normalized1 = []
    results_normalized2 = []
    if not no_parents:
        results_normalized1= list(preprocessing.scale(results_parents))

    if not no_co:
        results_normalized2= list(preprocessing.scale(results_co))

    results_substring = set(results_substring)

    # results_normalized = results_normalized1 + results_normalized2
    #
    # pairs_all = []
    # results_all = []
    # for i, element in enumerate(pairs_parents):
    #     if element in pairs_co:
    #         results_all.append((results_normalized[i] + results_normalized[len(results_parents) + pairs_co.index(element)]) / 2)
    #         pairs_all.append(element)
    #     else:
    #         results_all.append(results_normalized[i])
    #         pairs_all.append(element)
    # for i, element in enumerate(pairs_co):
    #     if element not in pairs_parents:
    #         results_all.append(results_normalized[len(results_parents) + i])
    #         pairs_all.append(element)
    #
    # new_relationships = list(find_outliers(results_all, pairs_all, threshold, mode = 'min'))
    #
    # outliers_parents  = set([])
    # outliers_co = set([])
    # #POINCARE
    outliers_parents = find_outliers(results_normalized1, pairs_parents, threshold, mode = 'min')
    #print(results_substring)
    new_relationships = list(outliers_parents|results_substring)
    # #CO_OCCURENCE
    # outliers_co = find_outliers(results_normalized2, pairs_co, threshold, mode = 'min')
    #
    # #outliers = list(outliers_parents.intersection(outliers_co))
    # #outliers = list(outliers_parents | outliers_co)
    # new_relationships = list(outliers_co)

    return new_relationships

def get_rank(current_child, parent, children, model, model_poincare, no_parents, no_co, compound  = True, wordnet = False):
    result_co = 0
    pair_co  = 0
    result_parent = 0
    pair_parent = 0
    current_child2  = current_child.replace(compound_operator, " ")
    parent2 = parent.replace(compound_operator, " ")
    if not no_co:
        try:
            children = [chi for chi in children if chi != current_child]
            if children:
                most_similar_child = model.wv.most_similar_to_given(current_child, children)
                index_child = model.wv.rank(current_child, most_similar_child)

                result_co = index_child
                pair_co = (current_child,parent)
            else:
                index_child = 0
        except (KeyError,ZeroDivisionError) as e:
            index_child = 0
    if not no_parents:
        try:
            if wordnet:
                node_senses = [n_sense.name() for n_sense in wn.synsets(current_child) if current_child in n_sense.name()]
                parent_senses = [p_sense.name() for p_sense in wn.synsets(parent) if parent in p_sense.name()]
                index_parent = 1000000
                for parent_sense in parent_senses:
                    for node_sense in node_senses:
                        index_parent_c = model_poincare.kv.rank(node_sense, parent_sense)
                        if index_parent_c < index_parent:
                            index_parent = index_parent_c
                if index_parent == 1000000:
                    index_parent = 0
            else:
                if compound:
                    index_parent = model_poincare.kv.rank(current_child, parent)
                else:
                    index_parent = model_poincare.kv.rank(current_child2,parent2)
                    # hierarchy_distance = model_poincare.kv.difference_in_hierarchy(child2, parent2)
                    # if hierarchy_distance >= 0:
                    #     index_parent = 0

            result_parent = index_parent
            pair_parent = (current_child,parent)


        except KeyError as e:
            index_parent = 0
    #print(result_parent)
    return [result_parent, pair_parent, result_co, pair_co]


#create dictionary mit den begirffen wegen bindestrich
def calculate_outliers(relations_o, model, model_poincare = None, threshold = None, no_parents = False, no_co = True, compound = False, wordnet = False):
    outliers = []
    structure = {}
    results_parents = []
    pairs_parents = []
    results_co = []
    pairs_co = []
    relations = relations_o.copy()
    for i in range(len(relations)):
        relations[i] = (relations[i][0].replace(" ", compound_operator), relations[i][1].replace(" ", compound_operator))

    for parent in [relation[1] for relation in relations]:
        structure[parent] = [relation[0] for relation in relations if relation[1] == parent]

    for key in structure:
        #print(key)
        if structure[key] == []:
            print("no children: " + key)
            continue
        elif not key in model.wv:
            continue
        cleaned_co_hyponyms = []
        for word in structure[key]:
            if word in model.wv:
                cleaned_co_hyponyms.append(word)
        if len(cleaned_co_hyponyms) < 1:
            continue


        cleaned_co_hyponyms_copy = cleaned_co_hyponyms.copy()
        for child in cleaned_co_hyponyms_copy:
            result_parent, pair_parent, result_co, pair_co = get_rank(child, key, cleaned_co_hyponyms, model, model_poincare, no_parents, no_co, compound, wordnet)
            if result_parent != 0 and child.split("_")[0] != key and child.split("_")[-1] != key:
                results_parents.append(result_parent)
                pairs_parents.append(pair_parent)
            if result_co != 0:
                results_co.append(result_co)
                pairs_co.append(pair_co)


    if not no_parents:
        results_normalized1= list(preprocessing.scale(results_parents))
        outliers_parents = find_outliers(results_normalized1, pairs_parents, threshold)
        outliers = list(outliers_parents)

    if not no_co:
        results_normalized2= list(preprocessing.scale(results_co))
        outliers_co = find_outliers(results_normalized2, pairs_co, threshold)
        outliers = list(outliers_parents)

    # results_normalized = results_normalized1 + results_normalized2
    # pairs_all = []
    # results_all = []
    # for i, element in enumerate(pairs_parents):
    #     if element in pairs_co:
    #         results_all.append((results_normalized[i] + results_normalized[len(results_parents) + pairs_co.index(element)]) / 2)
    #         pairs_all.append(element)
    #     else:
    #         results_all.append(results_normalized[i])
    #         pairs_all.append(element)
    # for i, element in enumerate(pairs_co):
    #     if element not in pairs_parents:
    #         results_all.append(results_normalized[len(results_parents) + i])
    #         pairs_all.append(element)
    # outliers = find_outliers(results_all, pairs_all, threshold)


    if not no_co and not no_parents:
        outliers = list(outliers_parents.intersection(outliers_co))
        #outliers = list(outliers_parents | outliers_co)
    #print(outliers)
    return outliers


def find_outliers(results, pairs, threshold, mode = "max"):
    outliers = set([])
    num_clusters = threshold#15 wordnet #6 own_poincare
    results_all_s = np.asarray(results).reshape(-1,1)
    kmeans = KMeans(n_clusters=num_clusters, random_state = 0).fit(results_all_s) #3 for wordnet
    pred = kmeans.predict(results_all_s)
    pred_common = Counter(pred).most_common(2)
    main_cluster,_ = pred_common[0]
    indices = []
    remaining = results.copy()
    for j in range(num_clusters):
        if mode == "max":
            result_max = max(remaining)
        if mode == "min":
            result_max = min(remaining)
        cluster_max = kmeans.predict(np.asarray([result_max]).reshape(1, -1))[0]
        if cluster_max == main_cluster:
            if mode == 'max':
                break
            if mode == 'min':
                for i, element in enumerate(pred):
                    if element == cluster_max:
                        #print(results_all[i])
                        indices.append(i)
                break
        for i, element in enumerate(pred):
            if element == cluster_max:
                #print(results_all[i])
                indices.append(i)
                remaining.remove(results[i])
    for index in indices:
        outliers.add(pairs[index])
    return outliers


def main():
    parser = argparse.ArgumentParser(description="Embeddings for Taxonomy")
    parser.add_argument('-m', '--mode', type=str, default='preload', choices=["combined_embeddings_removal_and_new", "combined_embeddings_new_nodes", "combined_embeddings_removal"], help="Mode of the system.")
    parser.add_argument('-d', '--domain', type=str, default='science', choices=["science", "food", "environment"], help="Domain")
    parser.add_argument('-e', '--embedding', type=str, nargs='?', default=None, choices=["own_and_poincare", "poincare", "poincare_all", "fasttext", "wiki2M", "wiki1M_subword", "own_w2v", "quick", "none"], help="Embedding to use")
    parser.add_argument('-ep', '--exparent', action='store_true', help='Exclude "parent" relations')
    parser.add_argument('-ico', '--inco', action='store_true', help='Include "co-hypernym relations')
    parser.add_argument('-com', '--compound', action='store_true', help='Includes compound word in outlier removal')
    parser.add_argument('-wn', '--wordnet', action ='store_true', help= 'Use Wordnet instead of own embeddings')
    parser.add_argument('--experiment_name', nargs='?', type=str, default=None, help="Name of the Experiment")
    parser.add_argument('--log', action='store_true', help="Logs taxonomy and results")
    args = parser.parse_args()
    print("Mode: ", args.mode)
    run(args.mode, args.domain, args.embedding, args.exparent, args.inco, args.compound, args.wordnet, args.experiment_name, args.log)


def run(mode, domain, embedding, exclude_parent = False, include_co = False, compound = False, wordnet = False,  experiment_name = None, log = False):
    if embedding == "fasttext":
        #model = gensim.models.KeyedVectors.load_word2vec_format('wiki-news-300d-1M-subword.vec', binary=False)
        model = gensim.models.FastText.load_fasttext_format('wiki.en.bin')
        #model = gensim.models.FastText.load_fasttext_format('crawl-300d-2M.vec')
    elif embedding == "wiki2M":
        #model = gensim.models.FastText.load_fasttext_format('crawl-300d-2M.vec','vec')
        model = gensim.models.KeyedVectors.load_word2vec_format('embeddings/crawl-300d-2M.vec', binary=False)
        #model.save("crawl-300d-2M.bin")
    elif embedding == "wiki1M_subword":
        model = gensim.models.KeyedVectors.load_word2vec_format('embeddings/wiki-news-300d-1M-subword.vec', binary=False)

    elif embedding == "own_w2v":
        model = gensim.models.KeyedVectors.load('embeddings/own_embeddings_w2v')

    elif embedding == "quick":
        model = gensim.models.KeyedVectors.load_word2vec_format('embeddings/crawl-300d-2M.vec', binary=False, limit = 50000)

    elif embedding == 'own_and_poincare':
        print("init")
        model = gensim.models.KeyedVectors.load('embeddings/own_embeddings_w2v_all') #n2 #all
        #model_poincare = PoincareModel.load('embeddings/embeddings_' + domain +'_crawl_poincare_3_50')
        #model_poincare = PoincareModel.load('embeddings/embeddings_science_crawl_merge_poincare_10_3_50_02')

        model_poincare = PoincareModel.load('embeddings/poincare_common_domains02_5_3_50')
        #model_poincare = PoincareModel.load('embeddings/embeddings_poincare_wordnet')

    gold = []
    relations = []
    taxonomy = []
    outliers = []
    exclude_co = not include_co

    if mode =='combined_embeddings_removal':
        #thresholds = [2,4,6,8,10,12,14]poincare and co-hyper testrun
        thresholds = [6]
        for value in thresholds:
            gold, relations = read_all_data(domain)
            outliers = calculate_outliers(relations, model, threshold = value, model_poincare = model_poincare, compound = compound, no_parents = exclude_parent, no_co = exclude_co, wordnet = wordnet)
            compare_to_gold(gold = gold, taxonomy = relations, outliers = outliers, model = model, log  = "logs/" + mode + "_" + embedding + "_" + str(value), write_file = "out/" + mode + "_" + embedding + "_" + str(value))


    elif mode == 'combined_embeddings_new_nodes':
        #thresholds = [2]
        thresholds = [2,4,6,8,10,12,14] #poincare testrun
        #thresholds = [12,14,18,20] #co-hyper testrun
        for value in thresholds:
            gold, relations = read_all_data(domain)
            new_nodes = connect_new_nodes(taxonomy = relations, gold = gold, model = model, model_poincare = model_poincare, threshold = value,  no_parents = exclude_parent, no_co = exclude_co, wordnet = wordnet)
            compare_to_gold(gold = gold, taxonomy = relations, model = model, model_poincare = model_poincare, new_nodes =  new_nodes)


    elif mode == 'combined_embeddings_removal_and_new':
        gold, relations = read_all_data(domain)
        new_nodes = connect_new_nodes(taxonomy = relations, gold = gold, model = model, model_poincare = model_poincare, threshold = 2,  no_parents = exclude_parent, no_co = exclude_co, wordnet = wordnet)
        outliers = calculate_outliers(relations, model, threshold = 6, model_poincare = model_poincare, compound = compound, no_parents = exclude_parent, no_co = exclude_co, wordnet = wordnet)
        relations1 = compare_to_gold(gold = gold, taxonomy = relations, model = model, model_poincare = model_poincare, new_nodes =  new_nodes)
        relations2 = compare_to_gold(gold = gold, taxonomy = relations, model = model, model_poincare = model_poincare, new_nodes =  new_nodes, outliers = outliers)
        outliers = calculate_outliers(relations1, model, threshold = 6, model_poincare = model_poincare, compound = compound, no_parents = exclude_parent, no_co = exclude_co, wordnet = wordnet)
        compare_to_gold(gold = gold, taxonomy = relations1, outliers = outliers, new_nodes = new_nodes, model = model, model_poincare = model_poincare)

if __name__ == '__main__':
    main()
