#!/usr/bin/env python2

import argparse
import csv
import os
import glob

def clean(args):
    print("Started cleaning.")

    for input_file_str in args.input_file:
        for input_file in glob.glob(input_file_str):
            print("Loading the dataset from file '%s'..." % input_file)
            count_processed_lines = 0
            count_wrote_lines = 0

            with open(input_file, "r") as f_in:
                with open(input_file + ".cleaned.csv", "w+") as f_out:
                    reader = csv.reader(f_in, delimiter=args.csv_delimiter)

                    for i, line in enumerate(reader):
                        count_processed_lines += 1

                        if str(line[args.filter_index]).strip().lower() in ['1', 'y', 'j', 'true', 'wahr']:
                            row = []

                            for c in args.columns:
                                row.append(line[int(c)])

                            f_out.write(args.csv_delimiter.join(row) + "\n")
                            count_wrote_lines += 1

            print("Wrote %s lines and by processing %s lines (%.2f%%)."
                  % (count_wrote_lines, count_processed_lines, float(count_wrote_lines) / float(count_processed_lines) * 100.0))

    print("Cleaning finished.")


def main():
    print("Clean HypeNet predicted files...")
    parser = argparse.ArgumentParser(description='Removes columns and rows of a HypeNet output file.')
    parser.add_argument('-i', '--input_file', nargs="+", required=True, help="CSV-file")

    parser.add_argument('--columns', nargs="+", required=True, help="Columns to keep")
    parser.add_argument('--filter_index', type=int, required=True, help="Column to filter for 'true' values")
    parser.add_argument('--csv_delimiter', default="\t", help="Column to filter for 'true' values")

    args = parser.parse_args()

    print("Starting with arguments:")
    for arg in sorted(vars(args)):
        if arg not in ['mode']:
            print("   %s: %s" % (arg, getattr(args, arg)))

    clean(args)
    print("FINISHED.")



if __name__ == '__main__':
    main()
