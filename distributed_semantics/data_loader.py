#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import os
import pandas
import logging
import gzip
import sys
import string
import gensim
from gensim.test.utils import datapath
punctuations = string.punctuation
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords



compound_operator = "_"

parser = None


def read_trial_data():
    all_info = []
    trial_dataset_fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data/relations.csv")
    with open(trial_dataset_fpath, 'r') as f:
        reader = csv.reader(f, delimiter = '\t')
        for i, line in enumerate(reader):
            #id, hyponym, hypernym, correct, source
            all_info.append((line[0], line[1], line[2], line[3], line[4]))
    correct_relations = []
    wrong_relations = []
    all_relations = []
    taxonomy = []
    for entry in all_info:
        if entry[4] in ["WN_plants.taxo", "WN_vehicles.taxo", "ontolearn_AI.taxo"]: #alternatively add negative co-hypo
            if entry[3] == "1":
                correct_relations.append((entry[0], entry[1], entry[2]))
                all_relations.append((entry[0], entry[1], entry[2]))
            else:
                wrong_relations.append((entry[0], entry[1], entry[2]))
                all_relations.append((entry[0], entry[1], entry[2]))

    for i in range(len(all_relations)):
        all_relations[i] = (all_relations[i][0], all_relations[i][1].replace(" ", compound_operator), all_relations[i][2].replace(" ", compound_operator))
    return [correct_relations, all_relations, taxonomy]


def read_input(input_file, vocabulary):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    logging.info("reading file {0}...this may take a while".format(input_file))
    colnames = ["id,", "text"]
    data = pandas.read_csv(input_file, names= colnames)
    text = data.text.tolist()
    freq = {}
    print("Number of Reviews: " + str(len(text)))
    for i in range(len(text)):
        line = text[i]
        if (i%10000==0):
            logging.info ("read {0} reviews".format (i))
            print(line)

        line = line.lower()
        for word_voc in vocabulary:
            if word_voc in line and word_voc != word_voc.replace(' ', compound_operator):
                #print(word_voc + " " + str(i))
                line = line.replace(word_voc, word_voc.replace(' ', compound_operator))
                # print(freq)
                #print(line)
            if word_voc.replace(' ', "-") in line:
                line = line.replace(word_voc.replace(' ', '-'), word_voc.replace(' ', compound_operator))
        cleared_line = gensim.utils.simple_preprocess (line, max_len = 80)
        yield cleared_line
    print(freq)


def replace_str_index(text,index=0,replacement=''):
    return '%s%s%s'%(text[:index],replacement,text[index+1:])

def spacy_tokenizer(sentence):
    tokens = parser(sentence, disable=['parser', 'tagger', 'ner', 'textcat'])
    tokens = [tok.lemma_.lower() for tok in tokens]
    #print(tokens)
    tokens = [tok for tok in tokens]
    sentence_norm = " ".join(tokens)
    return sentence_norm

def adjust_input(target_word, vocabulary):
    target_original = target_word
    if target_word in vocabulary:
        return target_word
    target_word = spacy_tokenizer(target_word)
    if target_word in vocabulary:
        return target_word
    else:
        return target_original

def create_relation_files(relations_all, output_file_name, min_freq):
    f_out = open("data/" + output_file_name, 'w')
    output_freqs = []
    output_rels_all = []
    for output in relations_all:
        output_relations = output[0]
        output_freq = output[1]
        output_rels_all.append(output_relations)
        output_freqs.append(output_freq)

    for i, output_rels in enumerate(output_rels_all):
        if i== len(output_rels) - 2:
            break
        for j, other_out in enumerate(output_rels_all):
            if i <= j:
                continue
            for k,entry1  in enumerate(output_rels):
                for l, entry2 in enumerate(other_out):
                    #print(entry1[1], entry1[0])
                    if (entry1[1], entry1[0]) == entry2:
                        print("Found contradicting entry: ", entry2)
                        if j == len(output_freqs) - 1:
                            other_out.remove(entry2)
                            print("Removed entry from commoncrawl")
                        else:
                            diff_freq = output_freqs[i][entry1] - output_freqs[j][entry2]
                            if diff_freq >= min_freq:
                                print("Freq_diff:", diff_freq, "therefore remove from other rel")
                                other_out.remove(entry2)
                            elif abs(diff_freq) >= min_freq:
                                print("freq_diff:", diff_freq, "therefore remove from current rel")
                                output_rels.remove(entry1)
                            else:
                                print("freq_diff:", diff_freq, "therefore remove both entries")
                                output_rels.remove(entry1)
                                other_out.remove(entry2)

    for relations in output_rels_all:
        for relation in relations:
            #print(relation)
            f_out.write(relation[0].replace(' ', compound_operator) + '\t' + relation[1].replace(' ', compound_operator) + '\n')
    f_out.close()

def process_rel_file(min_freq, input_file, vocabulary):

    relations =  []
    relations_with_freq = {}
    filename_in = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/"  + input_file)
    with open(filename_in, 'r') as f:
        reader = csv.reader(f, delimiter = '\t')
        f.readline()
        for i, line in enumerate(reader):
            freq = int(line[2])
            #remove reflexiv and noise relations
            hyponym = adjust_input(line[0], vocabulary)
            hypernym = adjust_input(line[1], vocabulary)
            valid = int(freq) >= min_freq  and line[0] != line[1] and len(line[0]) > 3 and len(line[1]) > 3 and (line[0] in vocabulary or line[1] in vocabulary)
            if valid:
                vocabulary.add(hyponym)
                vocabulary.add(hypernym)
                #remove symmetric relations
                if (hypernym, hyponym) in relations:
                    freq_sym = relations_with_freq[(hypernym, hyponym)]
                    if freq > freq_sym:
                        relations.remove((hypernym, hyponym))
                        #print(hypernym, hyponym)
                        if freq - freq_sym > min_freq:
                            relations.append((hyponym, hypernym))
                            relations_with_freq[(hyponym,hypernym)] =  freq
                            #print(hypernym,hyponym)
                    else:
                        continue
                else:
                    relations.append((hyponym, hypernym))
                    relations_with_freq[(hyponym,hypernym)] =  freq
            # if line[0] != hyponym or line[1] != hypernym:
            #     print(line[0], line[1])
            #     print(hyponym, hypernym)
            #     print('\n')
    print(len(relations))
    return relations, relations_with_freq





def read_all_data(domain = 'science'):
    global compound_operator
    if domain == 'science':
        filename_in = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../out/science_en.csv-relations.csv-taxo-knn1.csv-pruned.csv-cleaned.csv")
        filename_gold = "data/gold_science.taxo"
    elif domain =='food':
        filename_in = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../out/food_en.csv-relations.csv-taxo-knn1.csv-pruned.csv-cleaned.csv")
        filename_gold = "data/gold_food.taxo"
    elif domain == 'environment':
        filename_in = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../out/environment_eurovoc_en.csv-relations.csv-taxo-knn1.csv-pruned.csv-cleaned.csv")
        filename_gold = "data/gold_environment.taxo"
    relations = []
    with open(filename_in, 'r') as f:
        reader = csv.reader(f, delimiter = '\t')
        for i, line in enumerate(reader):
            relations.append(( line[1], line[2]))

    gold= []
    with open(filename_gold, 'r') as f:
        reader = csv.reader(f, delimiter = '\t')
        for i, line in enumerate(reader):
            gold.append((line[1], line[2]))
    return [gold, relations]

def create_with_freq(freqs, input_name, domain):
    freq = [100, 500, 1000, 5000, 10000, 20000]
    gold, relations = read_all_data(domain)
    gold = set([relation[1] for relation in gold] + [relation[2] for relation in gold])
    for entry in freq:
        create_rel_file(entry, input_name, input_name + str(entry) + ".tsv", gold)

if __name__ == '__main__':
    import spacy

    parser = spacy.load('en_core_web_sm')
    if len(sys.argv) >= 2:
        mode = sys.argv[1]

    if mode == 'science_crawl':
        freq = [3,5]
        create_with_freq(freq, "en_science.csv", "science")

    elif mode == 'food_crawl':
        freq = [3,5]
        create_with_freq(freq, "en_food.csv", "food")

    elif mode == 'environment_crawl':
        freq = [3,5]
        create_with_freq(freq, "en_environment.csv", "environment")

    elif mode == 'common_and_domain':
        freq_common = 5
        freq_domain = 3
        all_vocabulary = []
        domains = ['science', 'food', 'environment']
        output_domains = []
        for domain in domains:
            gold, relations = read_all_data(domain)
            gold = set([relation[1] for relation in gold] + [relation[2] for relation in gold])
            all_vocabulary += gold
            output_domains.append(process_rel_file(freq_domain,"en_" + domain + ".csv" ,gold))
        output_domains.append(process_rel_file(freq_common, "en_ps59g.csv", set(all_vocabulary)))
        create_relation_files(output_domains,"poincare_common_domains.tsv",freq_common)


    # elif mode == 'common_and_science':
    #     print("hello")
    #     freq_common = [10, 50, 100, 500]
    #     freq_science = [3,5,10]
    #     for entry in freq_common:
    #         for entry_science in freq_science:
    #             create_rel_file([entry, entry_science], ["en_ps59g.csv", "en_science.csv"], "science_crawl_merge_" + str(entry) + "_" + str(entry_science) + ".tsv")
    # elif mode == 'merge_files':
    #     relations_science = []
    #     relations_common = []
    #     science_words = set([])
    #     filename_in_science = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/"  + "science_crawl_3.tsv")
    #     with open(filename_in_science, 'r') as f:
    #         reader = csv.reader(f, delimiter = '\t')
    #         f.readline()
    #         for i, line in enumerate(reader):
    #             relations_science.append((line[0], line[1]))
    #             science_words.add(line[0])
    #             science_words.add(line[1])
    #
    #     filename_in = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/commoncrawl_small_10.tsv")
    #     with open(filename_in, 'r') as f:
    #         reader = csv.reader(f, delimiter = '\t')
    #         f.readline()
    #         for i, line in enumerate(reader):
    #             if line[0] not in science_words and line[1] not in science_words:
    #                 relations_common.append((line[0], line[1]))
    #     relations_merged = set(relations_science + relations_common)
    #     #print(relations_merged)
    #     filename_out = open("data/" + "science_crawl_merge_10_3_02.tsv", 'w')
    #     for relation in relations_merged:
    #         filename_out.write(relation[0] + '\t' + relation[1] + '\n')
    #     filename_out.close()
