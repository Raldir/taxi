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
import operator
from nltk.corpus import wordnet as wn
py.sign_in('RamiA', 'lAA8oTL51miiC79o3Hrz')
# from spacy.en import English
# parser = spacy.load('en_core_web_md')
import pandas


def compare_to_gold(gold, taxonomy, model,  model_poincare = None, outliers = [], add_back = False, threshold_add = 0.4, new_nodes = [], log = "", write_file = ""):
    taxonomy_c = taxonomy.copy()
    global compound_operator
    removed_outliers = []
    for element in taxonomy_c:
        if (element[1].replace(' ', compound_operator), element[2].replace(' ', compound_operator)) in outliers:
            #print("skip: " + element[1] + " " + element[2])
            if add_back:
                #element_f = element[1].replace(' ', compound_operator)
                element_f = element[1].replace(' ', compound_operator)
                best_parent, lowest_distance = connect_to_taxonomy(taxonomy_c.copy(),element_f, model, model_poincare)
                if lowest_distance < threshold:
                    print(element[1], best_parent)
                    removed_outliers.append((element[1], best_parent.replace(compound_operator, ' ')))

                # elif rank_root == "None":
                #     removed_outliers.append(element)
            continue
        removed_outliers.append((element[1], element[2]))
    if new_nodes:
        for element in new_nodes:
            removed_outliers.append(element)

    correct = 0
    for element in removed_outliers:
        for ele_g in gold:
            if element[0] == ele_g[1] and element[1] == ele_g[2]:
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
    print(str(recall).replace(".", ',') +'\t' + str(precision).replace(".", ',') + '\t' + str(2*precision *recall / (precision + recall)).replace(".", ','))
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
def connect_to_taxonomy(relations_o, relation_nodes, current_word, model, model_poincare, co_hypo_relevance = 40, threshold = 2):
    #print("The current word is: ", current_word)
    global compound_operator

    best_parent = None
    structure = {}

    for parent in relation_nodes:
        if parent == current_word:
            continue
        structure[parent] = [relation[1] for relation in relations_o if relation[2] == parent]

    lowest_distance = 100000
    for parent in structure:

        child_distance = 0
        len_c = len(structure[parent])

        try:
            children = [child.replace(" ", compound_operator) for child in structure[parent] if child.replace(" ", compound_operator) in model.wv.vocab]
            children = [child for child in children if child != current_word]
            if not children:
                index_child = 0
            else:
                # for child in children:
                #     child_distance += model.wv.distance(current_word, child)
                most_similar_child = model.wv.most_similar_to_given(current_word, children)
                child_distance = model.wv.distance(current_word, most_similar_child)
            #     index_child = child_distance / len(children)
            index_child = child_distance * 4
        except KeyError as e:
            index_child = 0


        #index_parent = get_parent_distances(current_word.replace(compound_operator, " "), model_poincare)
        try:
            #index_parent = model_poincare.kv.distance(current_word.replace(compound_operator, " "), parent)

            index_parent = model_poincare.kv.distance(current_word.replace(compound_operator, " "), parent)
            hierarchy_distance = model_poincare.kv.difference_in_hierarchy(current_word.replace(compound_operator, " "), parent)
            if hierarchy_distance >= 0:
                index_parent = 1000


        except:
            index_parent = 0

        combined_distance =  index_parent
        if combined_distance == 0:
            continue
        #print(current_word, parent, index_child, index_parent)
        if combined_distance == lowest_distance:
            height_curr = 0
            last_parent = None
            new_parent = parent
            hypo = [relation[1] for relation in relations_o]
            while new_parent != last_parent:
                last_parent = new_parent
                if last_parent in hypo:
                    index = hypo.index(last_parent)
                    new_parent = relations_o[index][2]
                    height_curr+=1

            height_old = 0
            last_parent = None
            new_parent = best_parent
            while new_parent != last_parent:
                last_parent = new_parent
                if last_parent in hypo:
                    index = hypo.index(last_parent)
                    new_parent = relations_o[index][2]
                    height_old+=1
            if height_old > height_curr:
                continue
            else:
                best_parent = parent
        if combined_distance < lowest_distance:
            #best_parent  = parent.replace( " ", compound_operator)
            best_parent = parent
            #print(best_parent)
            #print("current best parent", best_parent, index_child, index_parent )
            lowest_distance = combined_distance
            #if same distance take more special parent


            #print(current_word, parent, index_child, index_parent)
    #print(lowest_distance)
    if best_parent == None:
        raise KeyError()
    if lowest_distance > threshold:
        raise KeyError()
    return [best_parent, lowest_distance]


def connect_new_nodes(gold, taxonomy, model, model_poincare, threshold):
    new_nodes = set([])
    new_relationships = []
    gold_nodes = [relation[1] for relation in gold] + [relation[2] for relation in gold]
    taxonomy_nodes = (set([relation[1] for relation in taxonomy] + [relation[2] for relation in taxonomy]))
    for element in gold_nodes:
        if element not in taxonomy_nodes:
            new_nodes.add(element)
    print(len(new_nodes))
    count = 0
    count_p = 0
    for node in new_nodes:
        if node.replace(" ", compound_operator) in model.wv:
            count+=1
    for node in new_nodes:
        if node in model_poincare.kv:
            count_p +=1
    print(count, "in embedding")
    print(count_p, "in poincare_embedding")

    for node in new_nodes:
        try:
            parent, distance = connect_to_taxonomy(taxonomy, taxonomy_nodes, node.replace(" ", compound_operator), model, model_poincare, threshold = threshold)
        except ValueError as e:
            #print(str(e))
            continue
        except KeyError as e:
            #print(str(e))
            continue
        #if distance < threshold:
        new_relationships.append((node, parent))
    print(new_relationships)
    print(len(new_relationships))
    return new_relationships




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

def get_parent_distances(word, model_poincare):
    element_distances = []
    if word not in model_poincare.kv.vocab:
        return None
    for element in model_poincare.kv.vocab:
        distance = model_poincare.kv.distance(word, element)
        hierarchy_distance = model_poincare.kv.difference_in_hierarchy(word, element)
        element_distances.append((element, distance, hierarchy_distance))
    distances_sorted = sorted(element_distances, key=operator.itemgetter(1))
    distances_parents_sorted = []
    for element in distances_sorted:
        if element[2] < 0:
            distances_parents_sorted.append((element[0], element[1]))
    return distances_parents_sorted

#create dictionary mit den begirffen wegen bindestrich
def calculate_outliers(relations_o, model, mode, embedding_type = None, threshold = None, co_hypo_relevance = 40, model_poincare = None):
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
        elif not key in model.wv:
            continue
        cleaned_co_hyponyms = []
        for word in structure[key]:
            if word in model.wv:
                cleaned_co_hyponyms.append(word)
        if len(cleaned_co_hyponyms) < 1:
            continue


        cleaned_co_hyponyms_copy = cleaned_co_hyponyms.copy()
            # data_base_word_name2 = data_base_word_name + ".n.01"
        # child2 = child + ".n.01"
        for child in cleaned_co_hyponyms_copy:
            child2  = child.replace(compound_operator, " ")
            data_base_word_name2 = data_base_word_name.replace(compound_operator, " ")

            # if data_base_word_name2 not in model_poincare.kv.vocab or outlier2 not in model_poincare.kv.vocab:
            #     break

            #parent_distances = get_parent_distances(outlier2, model_poincare)
            try:
                children = [chi for chi in cleaned_co_hyponyms if chi != child]
                child_distance = 0
                for child_u in children:
                    #most_similar_child = model.wv.most_similar_to_given(child, children
                    child_distance += model.wv.distance(child, child_u)
                index_child = child_distance * 8 / len(children)
            except (KeyError,ZeroDivisionError) as e:
                #print(str(e))
                index_child = 0

            try:
                index_parent = model_poincare.kv.distance(child2, data_base_word_name2)
                hierarchy_distance = model_poincare.kv.difference_in_hierarchy(child2, data_base_word_name2)
                if hierarchy_distance >= 0:
                    index_parent = 1
            except KeyError:
                index_parent = 0

            index_combined = index_child + index_parent

            #print(child, key, index_child)

            if index_combined == 0:
                continue
            if index_combined > threshold:

                outliers.append((child, key))
                #print(child, key, index_parent, index_child)
                cleaned_co_hyponyms.remove(child)
                if len(cleaned_co_hyponyms) < 1:
                    #print(str(sim) + '\n')
                    break

    print(outliers)
    return outliers




def main():
    parser = argparse.ArgumentParser(description="Embeddings for Taxonomy")
    parser.add_argument('mode', type=str, default='preload', choices=["combined_embeddings_new_nodes","combined_embeddings_removal_add", "combined_embeddings_removal","train_poincare", "analysis", "visualize_embedding", "visualize_embedding_poincare", "normal", "train_word2vec", "gridsearch_removal", "gridsearch_removal_add", "gridsearch_removal_add_iterative"], help="Mode of the system.")
    parser.add_argument('domain', type=str, default='science', choices=["science", "food", "environment"], help="Mode of the system.")
    parser.add_argument('embedding', type=str, nargs='?', default=None, choices=["own_and_poincare", "poincare", "poincare_all", "fasttext", "wiki2M", "wiki1M_subword", "own_w2v", "quick", "none"], help="Classifier architecture of the system.")
    parser.add_argument('embedding_name', type=str, nargs='?', default=None, help="Classifier architecture of the system.")
    parser.add_argument('experiment_name', nargs='?', type=str, default=None, help="Name of the Experiment")
    parser.add_argument('--log', action='store_true', help="Logs taxonomy and results")
    parser.add_argument('--trial', action='store_true', help="Uses trial dataset")
    args = parser.parse_args()
    print("Mode: ", args.mode)
    run(args.mode, args.domain, args.embedding, args.embedding_name, args.experiment_name, args.log, args.trial)


def run(mode, domain, embedding, embedding_name, experiment_name = None, log = False, trial = False):
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
        model = PoincareModel.load('embeddings/' + embedding_name)
        print(len(model.kv.vocab))
        words = ["computer science", "biology", "physics", "science", "virology", "life science", "chemistry", "earth science", "algebra", "economics", "optics" "immunology"]
        for word in words:
            print("Current word: ", word)
            element_distances = []
            for element in model.kv.vocab:
                distance = model.kv.distance(word, element)
                hierarchy_distance = model.kv.difference_in_hierarchy(word, element)
                element_distances.append((element, distance, hierarchy_distance))
            distances_sorted = sorted(element_distances, key=operator.itemgetter(1))
            distances_parents_sorted = []
            for element in distances_sorted:
                if element[2] < 0:
                    distances_parents_sorted.append(element[0])
            print(distances_parents_sorted)

            if word in model.kv.vocab:
                print("Closest Parent: ", model.kv.closest_parent(word))
                print("Closest Child ", model.kv.closest_child(word))
                print("Descendants: ", model.kv.descendants(word))
                print("Ancestors: ", model.kv.ancestors(word))
                print("Hierarchy diff to Science: ", model.kv.difference_in_hierarchy(word, "science"))
                print('\n')
            else:
                print("Word not in Vocab")


    elif embedding == "poincare_all":
        #models = ["embeddings_science_crawl_poincare_10_50", "embeddings_science_crawl_poincare_3_50", "embeddings_science_crawl_poincare_5_50" ]
        models = [
        "embeddings_common_small_poincare_100_50", "embeddings_common_small_poincare_10_50", "embeddings_common_small_poincare_3_50", "embeddings_common_small_poincare_500_50", "embeddings_common_small_poincare_5_50",
        "embeddings_science_crawl_combined_common_poincare_10_10_50", "embeddings_science_crawl_combined_common_poincare_10_3_50", "embeddings_science_crawl_combined_common_poincare_10_5_50",
        "embeddings_science_crawl_combined_common_poincare_100_10_50",
        "embeddings_science_crawl_combined_common_poincare_100_3_50", "embeddings_science_crawl_combined_common_poincare_100_5_50", "embeddings_science_crawl_combined_common_poincare_500_10_50", "embeddings_science_crawl_combined_common_poincare_500_3_50"
        ,"embeddings_science_crawl_combined_common_poincare_500_5_50","embeddings_science_crawl_combined_common_poincare_50_10_50","embeddings_science_crawl_combined_common_poincare_50_3_50","embeddings_science_crawl_combined_common_poincare_50_5_50",
        "embeddings_science_crawl_poincare_10_50", "embeddings_science_crawl_poincare_3_50", "embeddings_science_crawl_poincare_5_50" ]
        #f = open(os.path.join("logs","poincare", "embedding_exemplary_results02.txt"), 'w', 1)
        gold, relations = read_all_data()
        all_relations = set([relation1[1] for relation1 in relations] + [relation2[2] for relation2 in relations])
        print("all relations occur", len(all_relations))
        for model_name in models:
            model = PoincareModel.load('embeddings/' + model_name)
            print("Model: ", model_name)
            print(len(model.kv.vocab))
            vectors = []
            names = []
            exist = 0
            for relation in all_relations:
                if relation in model.kv.vocab:
                    exist +=1
            print("Occurence in relations: ", exist)
            continue
            #f.write("Model: " + model_name + '\n')
            #f.write(str(len(model.kv.vocab)) + '\n')

            #print([word for word in model.kv.vocab.items()][:1])
            words = ["computer science", "biology", "physics", "science", "virology", "life science", "chemistry", "earth science", "algebra", "economics", "optics" "immunology"]
            for word in words:
                try:
                    print("Current word: ", word)
                    f.write("Current word: " + word + '\n')
                    if word in model.kv.vocab:
                        parent =  model.kv.closest_parent(word)
                        child = model.kv.closest_child(word)
                        if parent != None:
                            print("Closest Parent: ", model.kv.closest_parent(word))
                        if child != None:
                            print("Closest Child ", model.kv.closest_child(word))
                        print("Descendants: ", model.kv.descendants(word))
                        print("Ancestors: ", model.kv.ancestors(word))
                        print("Hierarchy diff to Science: ", model.kv.difference_in_hierarchy(word, "science"))
                        print('\n')
                        if parent != None:
                            f.write("Closest Parent: " + model.kv.closest_parent(word) + '\n')
                        else:
                            f.write("There is no parent for " + word)
                        if child != None:
                            f.write("Closest Child "+ str(model.kv.closest_child(word)) + '\n')
                        else:
                            f.write("There is no child for " + word)
                        f.write("Descendants: " + str(model.kv.descendants(word)) + '\n')
                        f.write("Ancestors: " +  str(model.kv.ancestors(word)) + '\n')
                        f.write("Hierarchy diff to Science: " + str(model.kv.difference_in_hierarchy(word, "science")) + '\n')
                        f.write('\n')
                    else:
                        print("Current word not in vocabulary")
                        f.write("Current word not in vocabulary \n")
                except:
                    f.write("Print error for word " + word)
                    print("Error at", word)

            print('\n')
            print('\n')
            #f.write('\n')
            #f.write('--------------------------------------------------------------------------------------\n')
    elif embedding == 'own_and_poincare':
        print("init")
        model = gensim.models.KeyedVectors.load('embeddings/own_embeddings_w2v_n2')
        model_poincare = PoincareModel.load('embeddings/embeddings_' + domain +'_crawl_poincare_3_50')
        #model_poincare = PoincareModel.load('embeddings/embeddings_science_crawl_merge_poincare_10_3_50_02')
        #model_poincare = PoincareModel.load('embeddings/embeddings_poincare_wordnet')

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
        gold,relations = read_all_data()
        freq_science = [3,5]
        for entry_science in freq_science:
            relations = './data/' + domain +'_crawl_' + str(entry_science) +'.tsv'
            #relations = './data/science_crawl_merge_10_3_02.tsv'
            poincare_rel = PoincareRelations(relations)
            dim = 50
            model = PoincareModel(poincare_rel, size = dim)
            print("Starting Training...")
            model.train(epochs=400)
            model.save("embeddings/embeddings_" + domain + "_crawl_poincare_" + str(entry_science) + "_" + str(dim))
            #model.save("embeddings/embeddings_science_crawl_merge_poincare_10_3_50_02")
            break

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
        gold, relations = read_all_data(domain)
        voc_rel = set([relation[1] for relation in relations] + [relation[2] for relation in relations])
        voc_gold = set([relation[1] for relation in gold] + [relation[2] for relation in gold])
        print("Vokabeln in Gold: " + str(len(voc_gold)) + "Vokabeln in Taxonomy: " + str(len(voc_rel)))

    if not trial:
        if mode == "normal":
            gold, relations = read_all_data(domain)
            for i in range(1,10):
                print(len(relations))
                outliers = calculate_outliers(relations, model, mode = "abs", embedding_type = embedding)
                relations = compare_to_gold(gold, relations,  outliers, model, write_file = "out/test")



        elif mode =="gridsearch_removal":
            threshholds = range(2,8, 1)
            threshholds = [float(value / 10) for value in threshholds]
            threshholds = [0.3, 0.32, 0.33, 0.35, 0.37,0.4]
            for value in threshholds:
                gold, relations = read_all_data(domain)
                outliers = calculate_outliers(relations,model, mode = "abs", embedding_type = embedding, threshhold=  value)
                compare_to_gold(gold, relations, outliers, model, mode = "removal", log  = "logs/" + experiment_name + "_" + str(value), write_file = "out/" + experiment_name + "_" + str(value))

        elif mode =="gridsearch_removal_add":
            threshholds = range(2,8)
            threshholds = [float(value / 10) for value in threshholds]
            threshholds = [0.3, 0.32, 0.33, 0.35, 0.37, 0.4]
            for value in threshholds:
                gold, relations = read_all_data(domain)
                outliers = calculate_outliers(relations,model, mode = "abs", embedding_type = embedding, threshhold=  value)
                compare_to_gold(gold, relations, outliers, model, mode = "removal_add", log  = "logs/" + experiment_name + "_" + str(value), write_file = "out/" + experiment_name + "_" + str(value))



        elif mode =="gridsearch_removal_add_iterative":
            threshholds = range(2, 5)
            threshholds = [float(value / 10) for value in threshholds]
            for value in threshholds:
                gold, relations  = read_all_data(domain)
                for i in range(3):
                    outliers = calculate_outliers(relations,model, "abs", embedding_type = embedding , threshhold = value)
                    relations = compare_to_gold(gold, relations, outliers, model, mode = "removal_add", log  = "logs/" + experiment_name + "_" + str(value), write_file = "out/" + experiment_name + "_" + str(value))

        elif mode =='combined_embeddings_removal':
            #thresholds = [1,5,10,20,30,50]
            #thresholds = [5, 50, 500, 1000] #wordnet

            #hresholds = [100, 50, 25, 15, 10, 5]
            thresholds = [8, 9, 9.4, 9.8,10]
            for value in thresholds:
                gold, relations = read_all_data(domain)
                outliers = calculate_outliers(relations, model, "abs", embedding_type = embedding, threshold = value, co_hypo_relevance = 25, model_poincare = model_poincare)
                compare_to_gold(gold = gold, taxonomy = relations, outliers = outliers, model = model, log  = "logs/" + mode + "_" + embedding + "_" + str(value), write_file = "out/" + mode + "_" + embedding + "_" + str(value))

        elif mode =='combined_embeddings_removal_add':
            #thresholds = [1,5,10,20,30,50]
            #thresholds = [5, 50, 500, 1000] #wordnet
            thresholds = [0.3, 0.5, 0.8, 1, 1.5, 2, 2.5]
            for value in thresholds:
                gold, relations = read_all_data(domain)
                #outliers = calculate_outliers(relations, model, "abs", embedding_type = embedding, threshold = 50, model_poincare = model_poincare)
                # outliers = [('climate', 'ecology'), ('meteorology', 'oceanography'), ('astrodynamics', 'music'),
                #  ('thermodynamics', 'chemical_engineering'), ('organic_chemistry', 'biochemistry'), ('genetics', 'biochemistry'),
                #   ('immunology', 'biochemistry'), ('probability', 'mathematics'), ('chemistry', 'mathematics'), ('composition', 'chemistry')]
                compare_to_gold(gold = gold, taxonomy = relations, outliers = outliers, model = model, model_poincare = model_poincare, add_back = True, log  = "logs/" + mode + "_" + embedding + "_" + str(value), write_file = "out/" + mode + "_" + embedding + "_" + str(value), threshold = value)
        elif mode =='combined_embeddings_new_nodes':
            thresholds = [1, 1.5, 1.75, 2, 2.5, 10000]
            for value in thresholds:
                gold, relations = read_all_data(domain)
                new_nodes = connect_new_nodes(taxonomy = relations, gold = gold, model = model, model_poincare = model_poincare, threshold = value)
                compare_to_gold(gold = gold, taxonomy = relations, model = model, model_poincare = model_poincare, new_nodes =  new_nodes)

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

# def create_compound_word(compound, model):
#     global compound_operator
#     word_parts = compound.split(compound_operator)
#     compound_word = np.copy(model.wv[word_parts[0]])
#     for i in range(1, len(word_parts)):
#         part = word_parts[i]
#         compound_word += model.wv[part]
#     compound_word /= (len(word_parts))
#     return compound_word
