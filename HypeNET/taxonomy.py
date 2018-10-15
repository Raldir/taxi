#!/usr/bin/env python2

import argparse
import csv
import random

from train.lstm_common import *
from train.paths_lstm_classifier import *
from common.knowledge_resource import KnowledgeResource
from sklearn.metrics import precision_recall_fscore_support

from itertools import count
from collections import defaultdict
from train.paths_lstm_classifier import PathLSTMClassifier

EMBEDDINGS_DIM = 50


def is_bool(x):
    return str(x).lower() in ("yes", "true", "t", "1")


def add_keys(corpus, dataset):
    print("Generate keys...")
    already_printed = set([])
    unknown_words = 0
    full_key = 0

    for (x, y) in dataset:
        id_x, id_y = corpus.get_id_by_term(str(x)), corpus.get_id_by_term(str(y))

        if id_x == -1 and x not in already_printed:
            unknown_words += 1
            print("Unknown word: %s" % x)

        if id_y == -1 and y not in already_printed:
            unknown_words += 1
            print("Unknown word: %s" % y)

        already_printed.add(x)
        already_printed.add(y)

        if id_x != -1 and id_y != -1:
            full_key += 1

        dataset[(x, y)]["keys"] = (id_x, id_y)

    print("Found %s (%.2f%%) known terms and %s (%.2f%%) unknown terms in given dataset with an input set of %s terms."
          % (len(already_printed) - unknown_words,
             float(len(already_printed) - unknown_words) / float(len(already_printed)) * 100.0,
             unknown_words,
             float(unknown_words) / float(len(already_printed)) * 100.0,
             len(already_printed)))

    print("Found %s full keysets of possible %s (%.2f%%)." % (
        full_key, len(dataset), float(full_key) / len(dataset) * 100.0))


def add_paths(corpus, dataset, lemma_index, pos_index, dep_index, dir_index):
    print("Generate paths...")

    for i, (x, y) in enumerate(dataset):
        keys = dataset[(x, y)]["keys"]
        vectorize_paths = {}

        try:
            paths = get_paths(corpus, keys[0], keys[1])
            dataset[(x, y)]["dependency_paths"] = paths

            for path, count in paths.iteritems():
                vectorized_path = vectorize_path(path, lemma_index, pos_index, dep_index, dir_index)

                if path is None:
                    print("   Path of %s / %s is none." % (x, y))
                elif vectorized_path is None:
                    print("   Vectorized path of %s / %s is none." % (x, y))
                else:
                    vectorize_paths[vectorized_path] = count
        except Exception as e:
            print("ERROR for pair %s / %s: %s" % (e, x, y))

        dataset[(x, y)]["paths"] = vectorize_paths

        if (i + 1) % (len(dataset) / 10) == 0:  # Print current state 10 times
            print("   %s / %s" % (i, len(dataset)))

    # keys = [(corpus.get_id_by_term(str(x)), corpus.get_id_by_term(str(y))) for (x, y) in dataset.keys()]
    # paths_x_to_y = [{vectorize_path(path, lemma_index, pos_index, dep_index, dir_index): count
    #                for path, count in get_paths(corpus, x_id, y_id).iteritems()}
    #               for (x_id, y_id) in keys]
    # paths_x_to_y = [{p: c for p, c in paths_x_to_y[i].iteritems() if p is not None} for i in range(len(keys))]

    print("Paths generated.")


def load_paths(dataset, lemma_index, pos_index, dep_index, dir_index, args):
    print("Load corpus from '%s' with prefix '%s'..." % (os.path.abspath(args.corpus_path), args.corpus_prefix))
    corpus = KnowledgeResource(args.corpus_path + args.corpus_prefix)
    print('Corpus loaded.')

    add_keys(corpus, dataset)
    add_paths(corpus, dataset, lemma_index, pos_index, dep_index, dir_index)

    clean_dataset = {}
    count_empty = 0

    for (x, y) in dataset:
        path_list = dataset[(x, y)]["paths"]

        if len(path_list.keys()) == 0:
            count_empty += 1

        if not args.only_with_paths or (path_list is not None and len(path_list.keys()) > 0):
            clean_dataset[(x, y)] = dataset[(x, y)]
            clean_dataset[(x, y)]["x_y_vectors"] = None \
                if args.path_based \
                else (lemma_index.get(x, 0), lemma_index.get(y, 0))

    print("Found %s pairs with and %s without paths in a dataset of %s pairs (%.2f%% found)." %
          (len(dataset) - count_empty, count_empty, len(dataset),
           float(len(dataset) - count_empty) / len(dataset) * 100.0))

    # Get the word embeddings for x and y (get a lemma index)
    # x_y_vectors = None \
    #    if args.path_based \
    #    else [(lemma_index.get(x, 0), lemma_index.get(y, 0)) for (x, y) in dataset
    #          if dataset[(x, y)]["index"] is not None]

    print('Done loading paths:')
    print('   Number of lemmas: %d' % len(lemma_index))
    print('   Number of pos tags: %d' % len(pos_index))
    print('   Number of dependency labels: %d' % len(dep_index))
    print('   Number of directions: %d' % len(dir_index))

    print("Returning %s entries of processed dataset." % len(clean_dataset))
    return clean_dataset


def training(args):
    print("Start training...")

    print('Loading the dataset...')
    train_set = load_dataset(args.dataset_path + 'train.tsv')
    test_set = load_dataset(args.dataset_path + 'test.tsv')
    val_set = load_dataset(args.dataset_path + 'val.tsv')
    # y_train = [1 if 'True' in train_set[key] else 0 for key in train_set.keys()]
    # y_test = [1 if 'True' in test_set[key] else 0 for key in test_set.keys()]
    # Uncomment if you'd like to load the validation set (e.g. to tune the hyper-parameters)
    # y_val = [1 if 'True' in val_set[key] else 0 for key in val_set.keys()]
    dataset = {}
    dataset.update({keys: {"data": 1 if is_bool(train_set[keys]) else 0, "type": "train_set"} for keys in train_set})
    dataset.update({keys: {"data": 1 if is_bool(test_set[keys]) else 0, "type": "test_set"} for keys in test_set})
    dataset.update({keys: {"data": 1 if is_bool(val_set[keys]) else 0, "type": "val_set"} for keys in val_set})
    print('Done loading dataset')

    print("Initializing word embeddings with file '%s'..." % os.path.abspath(args.embeddings_file))
    wv, lemma_index = load_embeddings(args.embeddings_file)
    print('Finished loading word embeddings.')

    print("Define the dictionaries.")
    pos_index = defaultdict(count(0).next)
    dep_index = defaultdict(count(0).next)
    dir_index = defaultdict(count(0).next)

    dummy = pos_index['#UNKNOWN#']
    dummy = dep_index['#UNKNOWN#']
    dummy = dir_index['#UNKNOWN#']

    print('Load the paths and create the feature vectors...')
    cleaned_dataset = load_paths(
        dataset,
        lemma_index,
        pos_index,
        dep_index,
        dir_index,
        args
    )
    print("Done loading paths and feature vectors.")

    print("Generate training/test set...")
    x_train = []
    y_train = []
    x_y_vectors_train = None if args.path_based else []
    x_test = []
    y_test = []
    x_y_vectors_test = None if args.path_based else []

    for (x, y) in cleaned_dataset:
        if cleaned_dataset[(x, y)]["paths"] is None or len(cleaned_dataset[(x, y)]["paths"]) == 0:
            print("   %s / %s has None-path." % (x, y))

        if cleaned_dataset[(x, y)]["data"] is None:
            print("   %s / %s has None-data." % (x, y))

        if cleaned_dataset[(x, y)]["type"] == "train_set":
            x_train.append(cleaned_dataset[(x, y)]["paths"])
            y_train.append(cleaned_dataset[(x, y)]["data"])

            if x_y_vectors_train is not None:
                x_y_vectors_train.append(cleaned_dataset[(x, y)]["x_y_vectors"])
        elif cleaned_dataset[(x, y)]["type"] == "test_set":
            x_test.append(cleaned_dataset[(x, y)]["paths"])
            y_test.append(cleaned_dataset[(x, y)]["data"])

            if x_y_vectors_test is not None:
                x_y_vectors_test.append(cleaned_dataset[(x, y)]["x_y_vectors"])

    # X_train = dataset_instances[:len(train_set)]
    # X_test = dataset_instances[len(train_set):len(train_set) + len(test_set)]
    # Uncomment if you'd like to load the validation set (e.g. to tune the hyper-parameters)
    # X_val = dataset_instances[len(train_set)+len(test_set):]

    # x_y_vectors_train = None if x_y_vectors is None else x_y_vectors[:len(train_set)]
    # x_y_vectors_test = None if x_y_vectors is None else x_y_vectors[len(train_set):len(train_set) + len(test_set)]
    # Uncomment if you'd like to load the validation set (e.g. to tune the hyper-parameters)
    # x_y_vectors_val = x_y_vectors[len(train_set)+len(test_set):]
    print("Training/test set generated.")
    print("   Training set: %s" % len(x_train))
    print("   Embeddings training: %s" % ("/" if x_y_vectors_train is None else str(len(x_y_vectors_train))))
    print("   Test set: %s" % len(x_test))
    print("   Embeddings test: %s" % ("/" if x_y_vectors_test is None else str(len(x_y_vectors_test))))

    print('Create the classifier...')
    classifier = PathLSTMClassifier(num_lemmas=len(lemma_index),
                                    num_pos=len(pos_index),
                                    num_dep=len(dep_index),
                                    num_directions=len(dir_index),
                                    n_epochs=args.epochs,
                                    num_relations=2,
                                    lemma_embeddings=wv,
                                    dropout=args.word_dropout_rate,
                                    alpha=args.alpha,
                                    use_xy_embeddings=not args.path_based)
    print('Classifier created.')

    print('Training with learning rate = %f, dropout = %f...' % (args.alpha, args.word_dropout_rate))
    classifier.fit(x_train, y_train, x_y_vectors=x_y_vectors_train)
    print('Classifier finished training.')

    if args.evaluate:
        print('Evaluation:')
        pred = classifier.predict(x_test, x_y_vectors=x_y_vectors_test)
        p, r, f1, support = precision_recall_fscore_support(y_test, pred, average='binary')
        print('Precision: %.3f, Recall: %.3f, F1: %.3f' % (p, r, f1))

    print("Training finished.")

    print("Save model to '%s' with prefix '%s'..." % (os.path.abspath(args.model_path), args.model_prefix))
    classifier.save_model(args.model_path + args.model_prefix, [lemma_index, pos_index, dep_index, dir_index])
    print("Model saved.")

    print("Training task finished.")


def prediction(args):
    print("Started prediction.")

    print("Loading the dataset from file '%s'..." % args.hype_file)
    test_set = {}
    data_field_length = 0
    with open(args.hype_file, "r") as f:
        reader = csv.reader(f, delimiter=args.csv_delimiter)

        for i, line in enumerate(reader):
            # CSV-reader does not have an option to skip a header line...
            if not args.csv_has_header or i > 0:
                key = line[args.csv_tuple_start_index:args.csv_tuple_start_index + 2]

                test_set[tuple(key)] = {
                    "data": line
                }
                data_field_length = len(line)

    print("Dummy dataset print: %s" % test_set.keys()[
                                      0: min(len(test_set.keys()), 5)])  # Print first 5 entries if possible
    print('Dataset loaded.')

    print("Load model from '%s' with prefix '%s'..." % (os.path.abspath(args.model_path), args.model_prefix))
    classifier, lemma_index, pos_index, dep_index, dir_index = load_model(args.model_path + args.model_prefix)
    print("Model loaded.")

    print('Load the paths and create the feature vectors...')
    cleaned_dataset = load_paths(
        test_set,
        lemma_index,
        pos_index,
        dep_index,
        dir_index,
        args
    )
    print("Done loading paths and feature vectors.")

    print("Generate dataset...")
    x_test = []
    x_y_vectors_test = None if args.path_based else []

    for keys in cleaned_dataset:
        cleaned_dataset[keys]["pred_index"] = len(x_test)
        x_test.append(cleaned_dataset[keys]["paths"])

        if x_y_vectors_test is not None:
            x_y_vectors_test.append(cleaned_dataset[keys]["x_y_vectors"])
    print("Done generating dataset.")

    print('Start prediction...')
    pred = classifier.predict(x_test, x_y_vectors=x_y_vectors_test, full_information=True)
    print('Prediction finished.')

    print("Write result to: %s" % os.path.abspath(args.output_file))
    c_printed_lines = 0
    c_positive = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    with open(args.output_file, "w+") as f:
        writer = csv.writer(f, delimiter=args.csv_delimiter)
        write_predictions = []

        if args.write_predictions == 'all' or args.write_predictions == 'false':
            write_predictions.append(0)

        if args.write_predictions == 'all' or args.write_predictions == 'true':
            write_predictions.append(1)

        if args.write_header:
            header = ["col" + str(i) for i in range(0, data_field_length)]

            if args.include_predictions:
                header.append("prediction")

            if args.include_scores:
                header.append("prediction_score")
                header.append("#paths")

            writer.writerow(header)

        for (x, y) in cleaned_dataset:
            pred_index = cleaned_dataset[(x, y)]["pred_index"]
            predicted = pred[pred_index][0]
            prediction_score = pred[pred_index][1]

            if predicted in write_predictions:
                result = cleaned_dataset[(x, y)]["data"]

                if args.include_predictions:
                    result.append(predicted)

                if args.include_scores:
                    result.append(prediction_score)
                    result.append(len(cleaned_dataset[(x, y)]["paths"]))

                if args.validation_label_index is not None:
                    # - 2 because of the hypo/hyper-tuple which; label is expected after relation tuple
                    expected = cleaned_dataset[(x, y)]["data"][args.validation_label_index]
                    c_positive += 1 if is_bool(expected) else 0
                    tp += 1 if is_bool(expected) and is_bool(predicted) else 0
                    tn += 1 if not is_bool(expected) and not is_bool(predicted) else 0
                    fp += 1 if not is_bool(expected) and is_bool(predicted) else 0
                    fn += 1 if is_bool(expected) and not is_bool(predicted) else 0

                writer.writerow(result)
                c_printed_lines += 1

    print("Finished writing result (%s lines)." % c_printed_lines)

    if args.validation_label_index is not None:
        # print("+----------+----------+----------+----------+")
        # print("|          | Positive | Negative |    Sum   |")
        # print("+----------+----------+----------+----------+")
        # print("| Positive |          |          |          |")
        # print("+----------+----------+----------+----------+")
        # print("| Negative |          |          |          |")
        # print("+----------+----------+----------+----------+")
        l_tp = len(str(tp))
        l_tn = len(str(tn))
        l_fp = len(str(fp))
        l_fn = len(str(fn))
        l_sum_p = len(str(tp + fp))
        l_sum_n = len(str(tn + fn))

        max_length = max(l_tp, l_tn, l_fp, l_fn, l_sum_p, l_sum_n, len("Positive"), len("true \ pred")) + 2
        table = "   "
        table += ("+" + ("-" * max_length)) * 4 + "+\n   "

        table += "| true \ pred" + (" " * (max_length - len("true \ pred") - 1)) \
                 + "| Positive" + (" " * (max_length - len("Positive") - 1)) \
                 + "| Negative" + (" " * (max_length - len("Negative") - 1)) \
                 + "| Sum" + (" " * (max_length - len("Sum") - 1)) \
                 + "|\n   "

        table += ("+" + ("-" * max_length)) * 4 + "+\n   "

        table += "| Positive" + (" " * (max_length - len("Positive") - 1)) \
                 + "| " + str(tp) + (" " * (max_length - l_tp - 1)) \
                 + "| " + str(fp) + (" " * (max_length - l_fp - 1)) \
                 + "| " + str(tp + fp) + (" " * (max_length - l_sum_p - 1)) \
                 + "|\n   "

        table += ("+" + ("-" * max_length)) * 4 + "+\n   "

        table += "| Negative" + (" " * (max_length - len("Negative") - 1)) \
                 + "| " + str(fn) + (" " * (max_length - l_fn - 1)) \
                 + "| " + str(tn) + (" " * (max_length - l_tn - 1)) \
                 + "| " + str(fn + tn) + (" " * (max_length - l_sum_n - 1)) \
                 + "|\n   "

        table += ("+" + ("-" * max_length)) * 4 + "+"

        div = lambda x, y: float(x) / float(y) * 100.0 if y else 0
        print('Validation:')
        print(table)
        print("")
        print("   TP: %s" % tp)
        print("   FN: %s" % fn)
        print("   TN: %s" % tn)
        print("   FP: %s" % fp)
        print("")
        print("   Positive labeled: %s (%.3f%%)" % (c_positive, div(c_positive, len(pred))))
        print("   Negative labeled: %s (%.3f%%)" % (len(pred) - c_positive, div(len(pred) - c_positive, len(pred))))
        print("")
        print("   Positive predicted values: %.3f%%" % div(tp, tp + fp))
        print("   Negative predicted values: %.3f%%" % div(tn, tn + fn))
        print("")
        print("   Recall   : %.3f%%" % div(tp, tp + fn))
        print("   Precision: %.3f%%" % div(tp, tp + fp))
        print("")
        print("   Error rate: %.3f%%" % div(fn + fp, tp + fp + fn + tn))
        print("   Accuracy  : %.3f%%" % div(tp + tn, tp + fp + fn + tn))

        print("Prediction task finished.")


def main():
    print("Started HypeNet for taxonomy generation...")
    script_path = os.path.dirname(os.path.realpath(__file__)) + "/"

    parser = argparse.ArgumentParser(description='Run HypeNet for taxonomy generation.')
    parser.add_argument('-c', '--corpus_path', required=True, help="Path the directory with the corpus files.")
    parser.add_argument('-cp', '--corpus_prefix', default="corpus",
                        help="Prefix of the corpus file e.g. corpus for corpus_... Default: corpus")

    parser.add_argument('-m', '--model_path', default=script_path + 'model/',
                        help="Path the directory with the model files. Default: %s." % script_path + 'model/')

    parser.add_argument('-mp', '--model_prefix', default='wiki_model',
                        help="Prefix of the model files. Default: wiki_model")

    parser.add_argument('--path_based', default=True, type=is_bool, help="Path based analyses. Default: True")
    parser.add_argument('--only_with_paths', action='store_true',
                        help="Throws out term-pairs with zero paths. Default: False")

    parser.add_argument('--seed', default=int(random.getrandbits(32)),
                        help="Seed for the numpy randomizer. Default: random integer value")
    parser.add_argument('--dynet-gpus', default=1, help="Dynet gpus. Default: 1")
    parser.add_argument('--dynet-mem', default=4096, help="Dynet memory. Default: 4096")
    parser.add_argument('--dynet-seed', default=int(random.getrandbits(32)),
                        help="Seed for the dynet randomizer. Default: random integer value")

    subparser = parser.add_subparsers(dest="mode", help='Mode')

    tp = subparser.add_parser("training", help="Train a model with a dataset.")
    tp.add_argument('-d', '--dataset_path', default=script_path + 'dataset/datasets/dataset_rnd/')
    tp.add_argument('-e', '--embeddings_file', default=script_path + 'embedding/glove.6B.50d.txt')
    tp.add_argument('--epochs', default=3)
    tp.add_argument('--alpha', default=0.001)
    tp.add_argument('--word_dropout_rate', default=0.5)
    tp.add_argument('--evaluate', default=True, type=is_bool, help='Calculate precision, recall and F1.')

    pp = subparser.add_parser("prediction", help="Use a trained model and predict hypernyms.")
    pp.add_argument('-i', '--hype_file', required=True, help="CSV-file containing hypo/hyper-relations.")
    pp.add_argument('-o', '--output_file', default=script_path + 'result.csv')
    pp.add_argument('--include_scores', default=True, type=is_bool,
                    help="Includes the prediction score as new column.")
    pp.add_argument('--include_predictions', default=True, type=is_bool,
                    help="Includes the prediction (e.g. 1 or 0) as new column.")
    pp.add_argument('--write_predictions', choices=['all', 'true', 'false'], default='all',
                    help="Only writes lines predicted as true/false or write all (default).")
    pp.add_argument('--write_header', default=True, type=is_bool,
                    help="Writes a header row to the output.")
    pp.add_argument('--csv_delimiter', default='\t')
    pp.add_argument('--csv_has_header', default=False, type=is_bool)
    pp.add_argument('--csv_tuple_start_index', default=0, type=int,
                    help="Sets the start index of hypo/hyper-relations in CSV-files (e.g. first column is an ID)")
    pp.add_argument('--validation_label_index', default=None, type=int,
                    help="If set, column in CSV-file containing label for hypo/hyper-relation will be validated.")

    args = parser.parse_args()
    print("Starting with arguments:")
    for arg in sorted(vars(args)):
        if arg not in ['mode']:
            print("   %s: %s" % (arg, getattr(args, arg)))

    sys.argv.insert(1, '--dynet-gpus')
    sys.argv.insert(2, str(args.dynet_gpus))
    sys.argv.insert(3, '--dynet-mem')
    sys.argv.insert(4, str(args.dynet_mem))
    sys.argv.insert(5, '--dynet-seed')
    sys.argv.insert(6, str(args.dynet_seed))

    np.random.seed(args.seed)

    try:
        os.mkdir(args.model_path)
    except:
        # Just ignore
        pass

    args.corpus_path = args.corpus_path + ("" if args.corpus_path.endswith("/") else "/")
    args.model_path = args.model_path + ("" if args.model_path.endswith("/") else "/")

    if args.mode == "training":
        args.dataset_path = args.dataset_path + ("" if args.dataset_path.endswith("/") else "/")
        training(args)
    elif args.mode == "prediction":
        prediction(args)
    else:
        print("Unknown mode '%s'." % args.mode)

    print("FINISHED.")


# predictions = prediction(args)
# save_predictions(args, predictions)


if __name__ == '__main__':
    main()
