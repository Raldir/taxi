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
        count_paths.py <parsed_wiki_path_file> <out_path_file> <out_frequent_path_file> <frequent_path_count> 

        <parsed_wiki_path_file> = the parsed wikipedia dump file
        <out_path_file> = Output paths file
        <out_frequent_path_file> = Output frequent paths file
        <frequent_path_count>  = If a paths occured equal or higher to this number it will be a frequent paths
    """)
    in_filename = args['<parsed_wiki_path_file>']
    out_path_filename = args['<out_path_file>']
    out_frequent_path_filename = args['<out_frequent_path_file>']
    frequent_path_count = int(args['<frequent_path_count>'])

    print("Start reading from file: %s" % in_filename)
    line_count = 0
    wrote_paths = 0
    wrote_frequent_paths = 0

    last_row = None
    count = 1

    with codecs.open(in_filename, 'r', 'utf-8') as in_file:
        with codecs.open(out_path_filename, 'w', 'utf-8') as out_path_file:
            with codecs.open(out_frequent_path_filename, 'w', 'utf-8') as out_frequent_path_file:

                # Example row: X/PROPN/nsubj>_publish/VERB/ROOT_<Y/NOUN/dobj
                for line in in_file:
                    current = line.strip()

                    if last_row == current:
                        count += 1
                    elif last_row != current and last_row is not None:
                        # Write new line
                        out_path_file.write("%s\t%s\n" % (last_row, count))
                        wrote_paths += 1

                        if count >= frequent_path_count:
                            out_frequent_path_file.write("%s\n" % last_row)
                            wrote_frequent_paths += 1

                        count = 1

                    last_row = current
                    line_count += 1

                    if line_count % 100000 == 0:
                        print("   Read %s lines." % line_count)

                # Write the last line to the files
                out_path_file.write("%s\t%s\n" % (last_row, count))

                if count >= frequent_path_count:
                    out_frequent_path_file.write("%s\t%s\n" % (last_row, count))

    print("Finished counting paths.")
    print("Wrote %s paths." % wrote_paths)
    print("Wrote %s frequent paths." % wrote_frequent_paths)


if __name__ == '__main__':
    main()
