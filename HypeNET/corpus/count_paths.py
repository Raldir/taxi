import codecs
import csv
from docopt import docopt

DELIMITER = '\t'


def main():
    """
    Creates a "knowledge resource" from triplets file
    """

    # Get the arguments
    args = docopt("""Counts the paths the Wikipedia dump and create a triplets file, each line is formatted as follows: X\t\Y\tpath

    Usage:
        count_paths.py <parsed_wikipedia_file> <out_path_file> <out_frequent_path_file> <frequent_path_count> 

        <parsed_wikipedia_file> = the parsed wikipedia dump file
        <out_path_file> = Output paths file
        <out_frequent_path_file> = Output frequent paths file
        <frequent_path_count>  = If a paths occured equal or higher to this number it will be a frequent paths
    """)

    in_file = args['<parsed_wikipedia_file>']
    out_path_filename = args['<out_path_file>']
    out_frequent_path_filename = args['<out_frequent_path_file>']
    frequent_path_count = int(args['<frequent_path_count>'])

    paths = {}

    print("Start reading from file: %s" % in_file)

    with codecs.open(in_file, 'r', 'utf-8') as csvfile:
        # Example row:
        # institute	titles	X/PROPN/nsubj>_publish/VERB/ROOT_<Y/NOUN/dobj
        for line in csvfile:
            row = line.split(DELIMITER)
            path = row[2]

            if path not in paths:
                paths[path] = 0

            paths[path] += 1

    print("Finished reading.")

    print("Write paths to file: %s" % out_path_filename)
    print("Write frequent paths (occured >= %s) to file: %s" % (frequent_path_count, out_frequent_path_filename))
    frequent_path_wrote = 0

    with codecs.open(out_path_filename, 'w', 'utf-8') as out_path_file:
        with codecs.open(out_frequent_path_filename, 'w', 'utf-8') as out_frequent_path_file:

            for path in paths:
                frequency = paths[path]

                out_path_file.write('%s\t%s\n' % (path, frequency))

                if frequency >= frequent_path_count:
                    out_frequent_path_file.write("%s\n" % path)
                    frequent_path_wrote += 1

    print("Wrote %s paths." % len(paths))
    print("Wrote %s frequent paths." % frequent_path_wrote)

if __name__ == '__main__':
    main()
