import argparse

from util import load_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', required=True, type=str,
                    help='The path to the output file')

    args = parser.parse_args()
    path = args.path
    
    lines = load_file(path)

    avg_length = len([w for l in lines for w in l])/len(lines)

    print("Average sentence length: {0}".format(avg_length))
    
    