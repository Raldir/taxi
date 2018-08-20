import csv
import os
import pandas
import logging
import gzip
import sys



compound_operator = "_"


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
                print(word_voc + " " + str(i))
                line = line.replace(word_voc, word_voc.replace(' ', compound_operator))
                # print(freq)
                #print(line)
            if word_voc.replace(' ', "-") in line:
                line = line.replace(word_voc.replace(' ', '-'), word_voc.replace(' ', compound_operator))
        cleared_line = gensim.utils.simple_preprocess (line, max_len = 30)
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

def read_all_data():
    global compound_operator
    filename_in = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../out/science_en.csv-relations.csv-taxo-knn1.csv-pruned.csv-cleaned.csv")
    filename_gold = "data/gold.taxo"
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
