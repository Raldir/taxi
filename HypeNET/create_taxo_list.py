#!/usr/bin/env python2

import argparse
import csv
import os


def print_term_list(args, words):
    with open(args.input_file + ".wordlist", "w+") as f_out_wl:
        print("Create wordlist in: %s" % (args.input_file + ".wordlist"))
        writer_wl = csv.writer(f_out_wl, delimiter=args.csv_delimiter)

        for i, w1 in enumerate(words):
            writer_wl.writerow([w1])

            if (i + 1) % (len(words) / 10) == 0:  # Print current state 10 times
                print("%s / %s printed." % (i, len(words)))



def print_taxo(args, words):
    with open(args.input_file + ".taxo", "w+") as f_out_taxo:
        print("Create term combination list in: %s" % (args.input_file + ".taxo"))
        writer_taxo = csv.writer(f_out_taxo, delimiter=args.csv_delimiter)

        for i, w1 in enumerate(words):
            if (i + 1) % (len(words) / 10) == 0:  # Print current state 10 times
                print("%s / %s printed." % (i, len(words)))

            for w2 in words:
                writer_taxo.writerow([w1, w2])


def get_words(args, words):
    with open(args.input_file, "r") as f_in:
        for i, line in enumerate(f_in):
            for c in str(line).strip().split(args.csv_delimiter):
                try:
                    float(c)
                except:
                    words.add(str(c).strip().lower())


def main():
    parser = argparse.ArgumentParser(description='Removes columns and rows of a HypeNet output file.')
    parser.add_argument('-i', '--input_file', required=True, help="CSV-file")
    parser.add_argument('--print_word_list', action="store_true")
    parser.add_argument('--csv_delimiter', default="\t")

    args = parser.parse_args()
    words = set([])

    print("Read terms...")
    get_words(args, words)
    print("Finished reading terms.")

    print("Sort term list...")
    words = sorted(words)
    print("Term list sorted.")

    if args.print_word_list:
        print("Print term list...")
        print_term_list(args, words)
        print("Term list printed.")

    print("Print terms...")
    print_taxo(args, words)
    print("Terms printed.")


if __name__ == '__main__':
    main()
