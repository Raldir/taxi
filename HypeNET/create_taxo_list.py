#!/usr/bin/env python2

import argparse
import csv
import os



def print_words(args, words):
    with open(args.input_file + ".wordlist", "w+") as f_out_wl:
        with open(args.input_file + ".taxo", "w+") as f_out_taxo:
            writer_wl = csv.writer(f_out_wl, delimiter=args.csv_delimiter)
            writer_taxo = csv.writer(f_out_taxo, delimiter=args.csv_delimiter)

            for w1 in words:
                writer_wl.writerow([w1])

                for w2 in words:
                    writer_taxo.writerow([w1, w2])



def get_words(args, words):

    with open(args.input_file, "r") as f_in:
        reader = csv.reader(f_in, delimiter=args.csv_delimiter)

        for i, line in enumerate(reader):
            for c in line:
                if not str(c).isdigit():
                    words.add(str(c))








def main():
    parser = argparse.ArgumentParser(description='Removes columns and rows of a HypeNet output file.')
    parser.add_argument('-i', '--input_file', required=True, help="CSV-file")
    parser.add_argument('--csv_delimiter', default="\t")

    args = parser.parse_args()

    words = set([])
    get_words(args, words)
    print_words(args, words)



if __name__ == '__main__':
    main()
