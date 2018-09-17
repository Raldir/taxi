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


def generate_keys(corpus, dataset_keys):
    print("Generate keys...")
    keys = []

    already_printed = set([])
    unknown_words = 0
    full_key = 0

    for (x, y) in dataset_keys:
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

        keys.append((id_x, id_y))

    print("Found %s (%.2f%%) known terms and %s (%.2f%%) unknown terms in corpus with an input set of %s terms."
          % (len(already_printed) - unknown_words,
             float(len(already_printed) - unknown_words) / float(len(already_printed)) * 100.0,
             unknown_words,
             float(unknown_words) / float(len(already_printed)) * 100.0,
             len(already_printed)))

    print("Found %s full keysets of possible %s (%.2f%%)." % (
        full_key, len(dataset_keys), float(full_key) / len(dataset_keys) * 100.0))

    return keys


def load_paths(corpus_path, corpus_prefix, dataset_keys, lemma_index, pos_index, dep_index, dir_index, path_based=True):
    """
    Override load_paths from lstm_common to include (x, y) vectors
    :param corpus_prefix:
    :param dataset_keys:
    :return:
    """
    print("Load corpus from '%s' with prefix '%s'..." % (os.path.abspath(corpus_path), corpus_prefix))
    corpus = KnowledgeResource(corpus_path + corpus_prefix)
    print('Corpus loaded.')

    keys = generate_keys(corpus, dataset_keys)
    # keys = [(corpus.get_id_by_term(str(x)), corpus.get_id_by_term(str(y))) for (x, y) in dataset_keys]

    print("Generate paths...")
    paths_x_to_y = [{vectorize_path(path, lemma_index, pos_index, dep_index, dir_index): count
                     for path, count in get_paths(corpus, x_id, y_id).iteritems()}
                    for (x_id, y_id) in keys]
    paths_x_to_y = [{p: c for p, c in paths_x_to_y[i].iteritems() if p is not None} for i in range(len(keys))]

    paths = paths_x_to_y

    empty = [dataset_keys[i] for i, path_list in enumerate(paths) if len(path_list.keys()) == 0]
    print("Pairs without paths: %s, all dataset: %s" % (len(empty), len(dataset_keys)))

    # Get the word embeddings for x and y (get a lemma index)
    x_y_vectors = None \
        if path_based \
        else [(lemma_index.get(x, 0), lemma_index.get(y, 0)) for (x, y) in dataset_keys]

    print('Done loading paths:')
    print('   Number of lemmas: %d' % len(lemma_index))
    print('   Number of pos tags: %d' % len(pos_index))
    print('   Number of dependency labels: %d' % len(dep_index))
    print('   Number of directions: %d' % len(dir_index))

    return x_y_vectors, paths


def training(args):
    print("Start training...")

    print('Loading the dataset...')
    train_set = load_dataset(args.dataset_path + 'train.tsv')
    test_set = load_dataset(args.dataset_path + 'test.tsv')
    val_set = load_dataset(args.dataset_path + 'val.tsv')
    y_train = [1 if 'True' in train_set[key] else 0 for key in train_set.keys()]
    y_test = [1 if 'True' in test_set[key] else 0 for key in test_set.keys()]
    # Uncomment if you'd like to load the validation set (e.g. to tune the hyper-parameters)
    # y_val = [1 if 'True' in val_set[key] else 0 for key in val_set.keys()]
    dataset_keys = train_set.keys() + test_set.keys() + val_set.keys()
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
    x_y_vectors, dataset_instances = load_paths(
        args.corpus_path,
        args.corpus_prefix,
        dataset_keys,
        lemma_index,
        pos_index,
        dep_index,
        dir_index,
        args.path_based
    )
    print("Done loading paths and feature vectors.")

    print("Generate training set...")
    X_train = dataset_instances[:len(train_set)]
    X_test = dataset_instances[len(train_set):len(train_set) + len(test_set)]
    # Uncomment if you'd like to load the validation set (e.g. to tune the hyper-parameters)
    # X_val = dataset_instances[len(train_set)+len(test_set):]

    x_y_vectors_train = None if x_y_vectors is None else x_y_vectors[:len(train_set)]
    x_y_vectors_test = None if x_y_vectors is None else x_y_vectors[len(train_set):len(train_set) + len(test_set)]
    # Uncomment if you'd like to load the validation set (e.g. to tune the hyper-parameters)
    # x_y_vectors_val = x_y_vectors[len(train_set)+len(test_set):]
    print("Training set generated.")

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
                                    use_xy_embeddings=x_y_vectors is not None)
    print('Classifier created.')

    print('Training with learning rate = %f, dropout = %f...' % (args.alpha, args.word_dropout_rate))
    classifier.fit(X_train, y_train, x_y_vectors=x_y_vectors_train)
    print('Classifier finished training.')

    if args.evaluate:
        print('Evaluation:')
        pred = classifier.predict(X_test, x_y_vectors=x_y_vectors_test)
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
    test_set_mapping = []
    with open(args.hype_file, "r") as f:
        reader = csv.reader(f, delimiter=args.csv_delimiter)

        for i, line in enumerate(reader):
            # CSV-reader does not have an option to skip a header line...
            if not args.csv_has_header or i > 0:
                key = line[args.csv_tuple_start_index:args.csv_tuple_start_index + 2]

                test_set[tuple(key)] = line
                test_set_mapping.append(tuple(key))

    print("Dummy dataset print: %s" % test_set.keys()[
                                      0: min(len(test_set.keys()), 5)])  # Print first 5 entries if possible
    print('Dataset loaded.')

    print("Load model from '%s' with prefix '%s'..." % (os.path.abspath(args.model_path), args.model_prefix))
    classifier, lemma_index, pos_index, dep_index, dir_index = load_model(args.model_path + args.model_prefix)
    print("Model loaded.")

    print('Load the paths and create the feature vectors...')
    x_y_vectors_test, X_test = load_paths(
        args.corpus_path,
        args.corpus_prefix,
        test_set.keys(),
        lemma_index,
        pos_index,
        dep_index,
        dir_index,
        args.path_based
    )
    print("Done loading paths and feature vectors.")

    print('Start prediction...')
    pred = classifier.predict(X_test, x_y_vectors=x_y_vectors_test, full_information=args.include_score)
    print('Prediction finished.')

    print("Write result to: %s" % os.path.abspath(args.output_file))

    with open(args.output_file, "w+") as f:
        writer = csv.writer(f, delimiter=args.csv_delimiter)

        for i, p in enumerate(pred):
            hyper_hypo = test_set_mapping[i]

            # if hyper_hypo[0] not in already_printed and corpus.get_id_by_term(hyper_hypo[0]) == -1:
            #     print("Unknown hyponym in line %s: %s" % (i, hyper_hypo[0]))
            #     already_printed.add(hyper_hypo[0])
            #
            # if hyper_hypo[1] not in already_printed and corpus.get_id_by_term(hyper_hypo[1]) == -1:
            #     print("Unknown hypernym in line %s: %s" % (i, hyper_hypo[1]))
            #     already_printed.add(hyper_hypo[1])

            # predicted = p[0]
            # prediction_score = p[1]

            # writer.writerow(test_set[hyper_hypo] + [predicted, prediction_score])
            writer.writerow(test_set[hyper_hypo] + list(p))

    print("Finished writing result.")

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

    parser.add_argument('--path_based', default=True, type=lambda x: x.lower() in ("yes", "true", "t", "1"),
                        help="Path based analyses. Default: true")

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
    tp.add_argument('--evaluate', default=True, type=lambda x: x.lower() in ("yes", "true", "t", "1"),
                    help='Calculate precision, recall and F1.')

    pp = subparser.add_parser("prediction", help="Use a trained model and predict hypernyms.")
    pp.add_argument('-i', '--hype_file', required=True, help="CSV-file containing hypo/hyper-relations.")
    pp.add_argument('-o', '--output_file', default=script_path + 'result.csv')
    pp.add_argument('--include_score', default=True, type=lambda x: x.lower() in ("yes", "true", "t", "1"),
                    help="Includes the prediction score.")
    pp.add_argument('--csv_delimiter', default='\t')
    pp.add_argument('--csv_has_header', default=False, type=lambda x: x.lower() in ("yes", "true", "t", "1"))
    pp.add_argument('--csv_tuple_start_index', default=0, type=int,
                    help="Sets the start index of hypo/hyper-relations in CSV-files (e.g. first column is an ID)")

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


# predictions = prediction(args)
# save_predictions(args, predictions)


if __name__ == '__main__':
    main()
