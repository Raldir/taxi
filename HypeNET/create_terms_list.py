#!/usr/bin/env python2

import argparse
import re

import codecs

def read_terms(args):
    print("Read terms list from file '%s'..." % args.input_file)
    result = []

    with codecs.open(args.input_file, 'r') as f_in:
        # Read the next paragraph
        for line in f_in:
            cleaned_line = line.strip().lower()\
                .replace("a ", "")\
                .replace("an ", "")\
                .replace("the ", "")\
                .replace("{ ", "")\
                .replace("} ", "")\
                .replace("( ", "")\
                .replace(") ", "")\
                .replace("\" ", "")\
                .replace("' ", "")\
                .replace(". ", "") \
                .replace(", ", "") \
                .replace("; ", "")

            result.append(cleaned_line)

    print("Terms list read.")
    return result


def find_terms(args, term_list, prev_result, terms):
    result = set([])
    new_terms = set([])

    for line in term_list:
        for term in terms:
            if re.match("^[a-zA-Z]* " + term + "$", line):
                new_terms = new_terms.union(set(line.split(" ")))
                new_terms.add(line)
                result.add(line)

    return new_terms, prev_result.union(result)


def print_list(args, terms):
    output_file = args.output_directory\
                  + ("" if args.output_directory.endswith("/") else "/")\
                  + args.start_word \
                  + "_" + str(args.iterations) \
                  + ".terms"

    print("Write terms list to file '%s'..." % output_file)

    with codecs.open(output_file, 'w') as f_out:
        # Read the next paragraph
        for term in terms:
            f_out.write(str(term) + "\n")

    print("Terms list written.")


def main():
    print("Started HypeNet for taxonomy generation...")
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', required=True)
    parser.add_argument('-w', '--start_word', required=True)
    parser.add_argument('-n', '--iterations', type=int, default=3)
    parser.add_argument('-o', '--output_directory')
    args = parser.parse_args()

    term_list = read_terms(args)

    terms = {args.start_word}
    new_terms = {args.start_word}
    for i in range(0, args.iterations):
        print("Start iteration %s / %s." % (i + 1, args.iterations))
        new_terms, terms = find_terms(args, term_list, terms, new_terms)
        print("Found %s terms." % len(terms))

    print("Found %s terms in a list of %s terms for the given word '%s'." % (len(terms), len(term_list), args.start_word))
    print_list(args, terms)
    print("FINISHED.")


if __name__ == '__main__':
    main()
