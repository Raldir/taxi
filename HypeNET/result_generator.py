#!/usr/bin/env python2
import argparse
import csv
import glob


def print_k_values(input_file, k, values):
    with open(input_file + ".top" + str(k), "w+") as f_in:
        writer = csv.writer(f_in, delimiter="\t")

        for i, line in enumerate(values):
            if i < k:
                writer.writerow([i] + list(line))
            else:
                break


def read_values_of_file(args, input_file):
    result = []

    with open(input_file, "r") as f_in:
        reader = csv.reader(f_in, delimiter=args.csv_delimiter)

        for i, line in enumerate(reader):
            # col0	col1	prediction	prediction_score	#paths
            if i > 0 and line[0] != line[1]:
                result.append((line[0], line[1], float(line[3])))

    return result


def read_values(args):
    print("Started parsing.")

    for input_file in glob.glob(args.input_file):
        print("Loading the dataset from file '%s'..." % input_file)
        values = read_values_of_file(args, input_file)
        values = sorted(values, key=lambda x: x[2], reverse=True)

        for k in args.top:
            print_k_values(input_file, k, values)


def main():
    parser = argparse.ArgumentParser(description='Run HypeNet for taxonomy generation.')
    parser.add_argument('-i', '--input_file', required=True, help="CSV-file")
    parser.add_argument('-top', nargs="+", type=int, default=[15, 25, 50, 100, 250, 500, 1000])
    parser.add_argument('--csv_delimiter', default="\t", help="Column to filter for 'true' values")

    args = parser.parse_args()

    print("Starting with arguments:")
    for arg in sorted(vars(args)):
        if arg not in ['mode']:
            print("   %s: %s" % (arg, getattr(args, arg)))

    read_values(args)

    print("FINISHED.")


# predictions = prediction(args)
# save_predictions(args, predictions)


if __name__ == '__main__':
    main()
