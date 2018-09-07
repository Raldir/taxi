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
py.sign_in('RamiA', 'lAA8oTL51miiC79o3Hrz')
# from spacy.en import English
# parser = spacy.load('en_core_web_md')
import pandas


def compare_to_gold(gold, taxonomy_o, outliers, model, mode = "removal", log = False, write_file = None):
    taxonomy = taxonomy_o.copy()
    global compound_operator
    removed_outliers = []
    for element in taxonomy:
        if (element[1].replace(' ', compound_operator), element[2].replace(' ', compound_operator)) in outliers:
            #print("skip: " + element[1] + " " + element[2])
            if mode == "removal_add":
                best_word, parent, rank, rank_inv, rank_root = connect_to_taxonomy(taxonomy.copy(),element[1].replace(' ', compound_operator), model)
                if rank != None and rank_inv != None:
                    rank_ref =  rank + rank_inv
                    if  not rank_ref > 150:
                        if rank_root != None and rank_root in range(rank_ref -20, rank_ref + 20):
                        #and rank_root in range(rank_ref -20, rank_ref + 20):
                            removed_outliers.append((element[0], element[1], parent.replace(compound_operator, ' ')))
                            print("Added :" + str(element[0]) + " " + element[1] + " " +  parent.replace(compound_operator, ' '))
                            print("Best Word: " + best_word + ", Rank:" + str(rank) + " Rank_Inv: " + str(rank_inv) + ", Rank Parent: " + str(rank_root))

                # elif rank_root == "None":
                #     removed_outliers.append(element)
            continue
        removed_outliers.append(element)

    correct = 0
    for element in removed_outliers:
        for ele_g in gold:
            if element[1] == ele_g[1] and element[2] == ele_g[2]:
                correct+=1
                break
    precision = correct / float(len(removed_outliers))
    recall = correct / float(len(gold))
    print(float(len(removed_outliers)))
    print(float(len(gold)))
    print("Correct: " + str(correct))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F1: " + str(2*precision *recall / (precision + recall)))
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
            for element in removed_outliers:
                f.write(element[0] + '\t' + element[1] + '\t' + element[2]  + '\n')
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

def connect_to_taxonomy(relations_o, current_word, model):
    relations = relations_o.copy()
    global compound_operator
    for i in range(len(relations)):
        relations[i] = (relations[i][0], relations[i][1].replace(" ", compound_operator), relations[i][2].replace(" ", compound_operator))
    words_o = [relation[2] for relation in relations] + [relation[1] for relation in relations]
    words_a = [relation[2] for relation in relations if relation[2] in model.wv] + [relation[1] for relation in relations if relation[1] in model.wv]
    #print("Original" + str(len(words_o)) + "Remaining: " + str(len(words_a)))
    words_a = list(set(words_a))
    best_word = None
    #print(current_word)
    if not current_word in model.wv:
        print("outlier word not found in voc")
        return
    words_a.remove(current_word)
    element = model.wv.most_similar_to_given(current_word, words_a)
    while get_parent(relations, element) == current_word:
        words_a.remove(element)
        element = model.wv.most_similar_to_given(current_word, words_a)
    #print(current_word + " " + element)
    #curr_rank = model.wv.closer_than(current_word, element)
    rank = get_rank(current_word, element, model, 100000)
    if rank != None:
        best_word = element
    rank_inv = get_rank(element, current_word, model, 100000)
    parent =  get_parent(relations, element)
    rank_root = get_rank(current_word, parent , model, 100000)

    #print("Rank :" + str(rank) + ", Rank_Iverse:"+ str(rank_inv) +  ", Rank root: " + str(rank_root) + ", highest similarity: " + best_word + " " + current_word + ", parent: " +  parent)
    return [best_word, parent, rank, rank_inv, rank_root]




def visualize_taxonomy(taxonomy_vectors, taxonomy_names, name):
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(taxonomy_vectors)
    #plt.figure(figsize=(7,7))
    plt.scatter(Y[:, 0], Y[:, 1])
    for label, x, y in zip(taxonomy_names, Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', fontsize = 1)
    plt.show()
    plt.savefig(os.path.join("vis", name), dpi = 2000)




def calculate_outliers(relations_o, model, mode, embedding_type = None, threshhold = None):
    relations = relations_o.copy()
    structure = {}
    outliers = []
    for i in range(len(relations)):
        relations[i] = (relations[i][0], relations[i][1].replace(" ", compound_operator), relations[i][2].replace(" ", compound_operator))

    for parent in [relation[2] for relation in relations]:
        structure[parent] = [relation[1] for relation in relations if relation[2] == parent]

    for key in structure:
        data_base_word_name = key
        if structure[key] == []:
            print("no children: " + key)
            continue
        # if not key in model.wv and key.title() in model.wv:
        #     print "Uppercase in Model: " + key.title()
        #     data_base_word_name = key.title()
        elif not key in model.wv:
            continue
        cleaned_co_hyponyms = []
        for word in structure[key]:
            if word in model.wv:
                cleaned_co_hyponyms.append(word)
        if len(cleaned_co_hyponyms) < 1:
            continue
        #print(cleaned_co_hyponyms)
        above_treshhold = False
        while not above_treshhold:
            outlier = model.wv.doesnt_match(cleaned_co_hyponyms)
            sim = model.wv.similarity(data_base_word_name, outlier)
            #print(key + " " + outlier)
            if threshhold == None:
                if embedding_type == "0":
                    threshhold = 0.35#0.51
                elif embedding_type == "1":
                    threshhold = 0.6 #0.20
                else:
                    threshhold = 0.5

            #
            threshhold_bool = False
            if mode == "k_nearest":
                threshhold_bool = model.wv.rank(key, outlier) > threshhold and model.wv.rank(outlier, key) > threshhold
            if mode == "abs":
                threshhold_bool = sim < threshhold
            if threshhold_bool:
                outliers.append((outlier, key))
                cleaned_co_hyponyms.remove(outlier)
                if len(cleaned_co_hyponyms) < 1:
                    #print(str(sim) + '\n')
                    break
            else:
                above_treshhold = True
            #print(str(sim) + '\n')
    print(outliers)
    return outliers




def main():
    parser = argparse.ArgumentParser(description="Embeddings for Taxonomy")
    parser.add_argument('mode', type=str, default='preload', choices=["train_poincare", "analysis", "visualize_embedding", "visualize_embedding_poincare", "normal", "train_word2vec", "gridsearch_removal", "gridsearch_removal_add", "gridsearch_removal_add_iterative"], help="Mode of the system.")
    parser.add_argument('embedding', type=str, nargs='?', default=None, choices=["poincare", "poincare_all", "fasttext", "wiki2M", "wiki1M_subword", "own_w2v", "quick", "none"], help="Classifier architecture of the system.")
    parser.add_argument('embedding_name', type=str, nargs='?', default=None, help="Classifier architecture of the system.")
    parser.add_argument('experiment_name', nargs='?', type=str, default=None, help="Name of the Experiment")
    parser.add_argument('--log', action='store_true', help="Logs taxonomy and results")
    parser.add_argument('--trial', action='store_true', help="Uses trial dataset")
    args = parser.parse_args()
    print("Mode: ", args.mode)
    run(args.mode, args.embedding, args.embedding_name, args.experiment_name, args.log, args.trial)


def run(mode, embedding, embedding_name, experiment_name = None, log = False, trial = False):
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
    elif embedding == "poincare":
        model = PoincareModel.load('embeddings/poincare_common_domains02_5_3_50')
        print(len(model.kv.vocab))
        words = ["computer_science", "biology", "physics", "science", "virology", "life_science", "chemistry", "earth_science", "algebra", "economics", "optics" "immunology"]
        for word in words:
            print("Current word: ", word)

            if word in model.kv.vocab:
                try:
                    print("Closest Parent: ", model.kv.closest_parent(word))
                    print("Closest Child ", model.kv.closest_child(word))
                    print("Descendants: ", model.kv.descendants(word))
                    print("Ancestors: ", model.kv.ancestors(word))
                    print("Hierarchy diff to Science: ", model.kv.difference_in_hierarchy(word, "science"))
                    print('\n')
                except:
                    continue
            else:
                print("Word not in Vocab")


    gold = []
    relations = []
    taxonomy = []
    outliers = []

    if mode == "visualize_embedding_poincare":
        relations =  set([])
        filename_in = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data/isas_1000.tsv")
        with open(filename_in, 'r') as f:
            reader = csv.reader(f, delimiter = '\t')
            for i, line in enumerate(reader):
                relations.add((line[0], line[1]))
        plot = poincare_2d_visualization(model, relations, experiment_name)
        py.image.save_as(plot,"vis/" + experiment_name + '.png')
        print("Starting visualization")


        #visualize_taxonomy(vectors, names)
#todo own file for train
    if mode == "visualize_embedding":
        gold, relations = read_all_data()
        vectors = []
        names = []
        for relation in ([relation1[1].replace(" ", "_") for relation1 in relations] + [relation2[2].replace(" ", "_") for relation2 in relations]):
            if relation not in names:
                if relation not in model.wv:
                    print(relation)
                    continue
                vectors.append(model.wv[relation])
                names.append(relation)
        visualize_taxonomy(vectors, names, experiment_name)

    if mode == 'train_poincare':
        # gold,relations = read_all_data()
        # freq_science = [3,5]
        # for entry_science in freq_science:
        #     relations = './data/' + domain +'_crawl_' + str(entry_science) +'.tsv'
        #     #relations = './data/science_crawl_merge_10_3_02.tsv'
        #     poincare_rel = PoincareRelations(relations)
        #     dim = 50
        #     model = PoincareModel(poincare_rel, size = dim)
        #     print("Starting Training...")
        #     model.train(epochs=400)
        #     model.save("embeddings/embeddings_" + domain + "_crawl_poincare_" + str(entry_science) + "_" + str(dim))
        #     #model.save("embeddings/embeddings_science_crawl_merge_poincare_10_3_50_02")
        #     break
        relations = './data/poincare_common_domains.tsv'
        #relations = './data/science_crawl_merge_10_3_02.tsv'
        poincare_rel = PoincareRelations(relations)
        dim = 50
        model = PoincareModel(poincare_rel, size = dim)
        print("Starting Training...")
        model.train(epochs=400)
        model.save("embeddings/poincare_common_domains_5_3" + "_" + str(dim))

    if mode == "train_word2vec":
        gold_s,relations_s = read_all_data("science")
        gold_e,relations_e = read_all_data("environment")
        gold_f,relations_f = read_all_data("food")
        vocabulary = set([relation[2] for relation in gold_s] + [relation[1] for relation in gold_s])
        vocabulary = vocabulary | set([relation[2] for relation in gold_f] + [relation[1] for relation in gold_f])
        vocabulary = vocabulary | set([relation[2] for relation in gold_e] + [relation[1] for relation in gold_e])
        documents = list(read_input("/srv/data/5aly/data_text/wikipedia_utf8_filtered_20pageviews.csv",vocabulary))
        model = gensim.models.Word2Vec(size= 300, window = 5, min_count = 5, workers = 30)
        model.build_vocab(documents)
        #model.train(documents, total_examples = len(documents), epochs=10)
        model.train(documents, total_examples=model.corpus_count, epochs=30)
        model.save("embeddings/own_embeddings_w2v_all")

    elif mode == "analysis":
        gold, relations = read_all_data()
        voc_rel = set([relation[1] for relation in relations] + [relation[2] for relation in relations])
        voc_gold = set([relation[1] for relation in gold] + [relation[2] for relation in gold])
        print("Vokabeln in Gold: " + str(len(voc_gold)) + "Vokabeln in Taxonomy: " + str(len(voc_rel)))

    if not trial:
        if mode == "normal":
            gold, relations = read_all_data()
            for i in range(1,10):
                print(len(relations))
                outliers = calculate_outliers(relations, model, mode = "abs", embedding_type = embedding)
                relations = compare_to_gold(gold, relations,  outliers, model, write_file = "out/test")



        elif mode =="gridsearch_removal":
            threshholds = range(2,8, 1)
            threshholds = [float(value / 10) for value in threshholds]
            threshholds = [0.3, 0.32, 0.33, 0.35, 0.37,0.4]
            for value in threshholds:
                gold, relations = read_all_data()
                outliers = calculate_outliers(relations,model, mode = "abs", embedding_type = embedding, threshhold=  value)
                compare_to_gold(gold, relations, outliers, model, mode = "removal", log  = "logs/" + experiment_name + "_" + str(value), write_file = "out/" + experiment_name + "_" + str(value))

        elif mode =="gridsearch_removal_add":
            threshholds = range(2,8)
            threshholds = [float(value / 10) for value in threshholds]
            threshholds = [0.3, 0.32, 0.33, 0.35, 0.37, 0.4]
            for value in threshholds:
                gold, relations = read_all_data()
                outliers = calculate_outliers(relations,model, mode = "abs", embedding_type = embedding, threshhold=  value)
                compare_to_gold(gold, relations, outliers, model, mode = "removal_add", log  = "logs/" + experiment_name + "_" + str(value), write_file = "out/" + experiment_name + "_" + str(value))



        elif mode =="gridsearch_removal_add_iterative":
            threshholds = range(2, 5)
            threshholds = [float(value / 10) for value in threshholds]
            for value in threshholds:
                gold, relations  = read_all_data()
                for i in range(3):
                    outliers = calculate_outliers(relations,model, "abs", embedding_type = embedding , threshhold = value)
                    relations = compare_to_gold(gold, relations, outliers, model, mode = "removal_add", log  = "logs/" + experiment_name + "_" + str(value), write_file = "out/" + experiment_name + "_" + str(value))


if __name__ == '__main__':
    main()


# def valid_words(relations, model):
#     global compound_operator
#     valid_words = set([])
#     compound_words = {}
#     for relation in relations:
#         issubset_1 = True
#         issubset_2 = True
#         for entry in relation[1].split(compound_operator):
#             if not entry in model.wv:
#                 issubset_1 = False
#                 break
#         if relation[1] in model.wv:
#             valid_words.add(relation[1])
#         elif issubset_1:
#             #print word
#             compound_word = create_compound_word(relation[1], model)
#             compound_words[relation[1]] = compound_word
#             valid_words.add(relation[1])
#
#         for entry in relation[2].split(compound_operator):
#             if not entry in model.wv:
#                 issubset_2 = False
#                 break
#
#         if relation[2] in model.wv:
#             valid_words.add(relation[2])
#         elif issubset_2:
#             #print word
#             compound_word = create_compound_word(relation[2], model)
#             compound_words[relation[2]] = compound_word
#             valid_words.add(relation[2])
#             #model.syn0.build_vocab([relation[2]], update=True)
#             model.syn0[relation[2]] = compound_word
#
#     return valid_words, compound_words

#"embeddings_common_small_poincare_100_50", "embeddings_common_small_poincare_10_50", "embeddings_common_small_poincare_3_50", "embeddings_common_small_poincare_500_50", "embeddings_common_small_poincare_5_50",
# "embeddings_science_crawl_combined_common_poincare_10_10_50", "embeddings_science_crawl_combined_common_poincare_10_3_50", "embeddings_science_crawl_combined_common_poincare_10_5_50",
# "embeddings_science_crawl_combined_common_poincare_100_10_50",
# "embeddings_science_crawl_combined_common_poincare_100_3_50", "embeddings_science_crawl_combined_common_poincare_100_5_50", "embeddings_science_crawl_combined_common_poincare_500_10_50", "embeddings_science_crawl_combined_common_poincare_500_3_50"
# ,"embeddings_science_crawl_combined_common_poincare_500_5_50","embeddings_science_crawl_combined_common_poincare_50_10_50","embeddings_science_crawl_combined_common_poincare_50_3_50","embeddings_science_crawl_combined_common_poincare_50_5_50",
# "embeddings_science_crawl_poincare_10_50", "embeddings_science_crawl_poincare_3_50", "embeddings_science_crawl_poincare_5_50"
