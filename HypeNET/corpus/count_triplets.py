import codecs
import csv
from docopt import docopt

DELIMITER = '\t'


def main():
    """
    Creates a "knowledge resource" from triplets file
    """

    # Get the arguments
    args = docopt("""Counts the triplets and create a new file, where each line is formatted as follows: X\t\Y\tpath\tcount

    Usage:
        count_triplets.py <in_triplet_file> <out_quadruple_file> 

        <in_triplet_file> = Input file
        <out_quadruple_file> = Output file
    """)
    in_filename = args['<in_triplet_file>']
    out_filename = args['<out_quadruple_file>']

    print("Start reading from file: %s" % in_filename)
    line_count = 0

    last_row = None
    count = 1
    triple_count = 0

    with codecs.open(in_filename, 'r') as in_file:
        with codecs.open(out_filename, 'w') as out_file:

                # Example row: 34048729        17063612        60913773        1
                for line in in_file:
                    current = line.strip()

                    if last_row == current:
                        count += 1
                    elif last_row != current and last_row is not None:
                        # Write new line
                        out_file.write("%s\t%s\n" % (last_row, count))
                        triple_count += 1

                        count = 1

                    last_row = current
                    line_count += 1

                    if line_count % 1000000 == 0:
                        print("   Read %s lines." % line_count)

    print("Finished counting %s triplets." % triple_count)


if __name__ == '__main__':
    main()
