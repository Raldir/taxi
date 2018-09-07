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




def main():
    parser = argparse.ArgumentParser(description="Embeddings for Taxonomy")
    parser.add_argument('mode', type=str, default='preload', choices=["train_poincare", "analysis", "visualize_embedding", "visualize_embedding_poincare", "train_word2vec"], help="Mode of the system.")
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
