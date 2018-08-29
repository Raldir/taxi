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

def replace_str_index(text,index=0,replacement=''):
    return '%s%s%s'%(text[:index],replacement,text[index+1:])

def spacy_tokenizer(sentence):
    tokens = parser(sentence)
    tokens = [tok.lemma_.lower() if tok.lemma_ != "-PRON-" else tok.lower_ for tok in tokens]
    #print(tokens)
    tokens = [tok for tok in tokens]
    sentence_norm = " ".join(tokens)
    return sentence_norm

def adjust_input(target_word, input_relations):
    target_original = target_word
    if target_word in input_relations:
        return target_word
    target_word = spacy_tokenizer(target_word)
    if target_word in input_relations:
        return target_word
    else:
        return target_original

    # for word in input_relations:
    #     max_diff = int(len(target_word) / 8)
    #     while max_diff > 0:
    #         for i, character in enumerate(word):
    #             if i + 1 > len(target_word):
    #                 target_word = target_word + character
    #                 max_diff -= 1
    #                 break
    #             if target_word[i] != word[i]:
    #                 target_word = replace_str_index(target_word,i, character)
    #                 max_diff -=1
    #                 break
    #             #print(target_word)
    #         if target_word == word:
    #             return target_word
    #         if len(target_word) > len(word):
    #             target_word = target_word[:-1]
    #             max_diff -=1
    #     target_word = target_original
    # return target_word


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


def create_rel_file(min_freqs, input_files, output_name, relationsfile):
    f_out = open("data/" + output_name, 'w')
    for i, input_file in enumerate(input_files):
        min_freq = min_freqs[i]
        relations =  []
        relations_with_freq = {}
        filename_in = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/"  + input_file)
        with open(filename_in, 'r') as f:
            reader = csv.reader(f, delimiter = '\t')
            f.readline()
            for i, line in enumerate(reader):
                freq = int(line[2])
                #remove reflexiv and noise relations
                valid = int(freq) > min_freq  and line[0] != line[1] and len(line[0]) > 3 and len(line[1]) > 3
                hyponym = adjust_input(line[0], relationsfile)
                hypernym = adjust_input(line[1], relationsfile)
                if valid:
                    #remove symmetric relations
                    if (hypernym, hyponym) in relations:
                        freq_sym = relations_with_freq[(hypernym, hyponym)]
                        if freq > freq_sym:
                            relations.remove((hypernym, hyponym))
                            print(hypernym, hyponym)
                            if freq - freq_sym > min_freq:
                                relations.append((hyponym, hypernym))
                                relations_with_freq[(hyponym,hypernym)] =  freq
                                print(hypernym,hyponym)
                        else:
                            continue
                    else:
                        relations.append((hyponym, hypernym))
                        relations_with_freq[(hyponym,hypernym)] =  freq
                if line[0] != hyponym or line[1] != hypernym:
                    print(line[0], line[1])
                    print(hyponym, hypernym)
                    print('\n')
        print(len(relations))
        for relation in relations:
            f_out.write(relation[0] + '\t' + relation[1] + '\n')
    f_out.close()




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
            relations.append((line[0], line[1], line[2]))

    gold= []
    with open(filename_gold, 'r') as f:
        reader = csv.reader(f, delimiter = '\t')
        for i, line in enumerate(reader):
            gold.append((line[0], line[1], line[2]))
    return [gold, relations]


if __name__ == '__main__':
    import spacy

    parser = spacy.load('en_core_web_sm')
    if len(sys.argv) >= 2:
        mode = sys.argv[1]
    if mode == 'commoncrawl':
        freq = [100, 500, 1000, 5000, 10000, 20000]
        for entry in freq:
            create_rel_file([entry], ["isas-commoncrawl.csv"], "isas_" + str(entry) + ".tsv")
    elif mode == 'smaller_commoncrawl':
        freq = [5, 10, 50, 100, 500]
        for entry in freq:
            create_rel_file([entry], ["en_ps59g.csv"], "commoncrawl_small_" + str(entry) + ".tsv")
    elif mode == 'science_crawl':
        freq = [3,5]
        gold, relations = read_all_data('science')
        gold = set([relation[1] for relation in gold] + [relation[2] for relation in gold])
        for entry in freq:
            create_rel_file([entry], ["en_science.csv"], "science_crawl_" + str(entry) + ".tsv", gold)

    elif mode == 'food_crawl':
        freq = [3,5]
        gold, relations = read_all_data('food')
        gold = set([relation[1] for relation in gold] + [relation[2] for relation in gold])
        # freq = [3,5,10]
        for entry in freq:
            create_rel_file([entry], ["en_food.csv"], "food_crawl_" + str(entry) + ".tsv", gold)

    elif mode == 'environment_crawl':
        freq = [3,5]
        gold, relations = read_all_data('environment')
        gold = set([relation[1] for relation in gold] + [relation[2] for relation in gold])
        # freq = [3,5,10]
        for entry in freq:
            create_rel_file([entry], ["en_environment.csv"], "environment_crawl_" + str(entry) + ".tsv", gold)

    elif mode == 'common_and_science':
        print("hello")
        freq_common = [10, 50, 100, 500]
        freq_science = [3,5,10]
        for entry in freq_common:
            for entry_science in freq_science:
                create_rel_file([entry, entry_science], ["en_ps59g.csv", "en_science.csv"], "science_crawl_merge_" + str(entry) + "_" + str(entry_science) + ".tsv")
    elif mode == 'all':
        gold, relations = read_all_data()
        gold = set([relation[1] for relation in gold] + [relation[2] for relation in gold])
        # freq = [3,5,10]
        # for entry in freq:
        #     create_rel_file([entry], ["en_science.csv"], "science_crawl_" + str(entry) + ".tsv", gold)
        freq = [10, 50, 100, 500]
        for entry in freq:
            create_rel_file([entry], ["en_ps59g.csv"], "commoncrawl_small_" + str(entry) + ".tsv", gold)
        freq_common = [10, 50, 100, 500]
        freq_science = [3,5,10]
        for entry in freq_common:
            for entry_science in freq_science:
                create_rel_file([entry, entry_science], ["en_ps59g.csv", "en_science.csv"], "science_crawl_merge_" + str(entry) + "_" + str(entry_science) + ".tsv", gold)

    elif mode == 'merge_files':
        relations_science = []
        relations_common = []
        science_words = set([])
        filename_in_science = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/"  + "science_crawl_3.tsv")
        with open(filename_in_science, 'r') as f:
            reader = csv.reader(f, delimiter = '\t')
            f.readline()
            for i, line in enumerate(reader):
                relations_science.append((line[0], line[1]))
                science_words.add(line[0])
                science_words.add(line[1])

        filename_in = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/commoncrawl_small_10.tsv")
        with open(filename_in, 'r') as f:
            reader = csv.reader(f, delimiter = '\t')
            f.readline()
            for i, line in enumerate(reader):
                if line[0] not in science_words and line[1] not in science_words:
                    relations_common.append((line[0], line[1]))
        relations_merged = set(relations_science + relations_common)
        #print(relations_merged)
        filename_out = open("data/" + "science_crawl_merge_10_3_02.tsv", 'w')
        for relation in relations_merged:
            filename_out.write(relation[0] + '\t' + relation[1] + '\n')
        filename_out.close()
