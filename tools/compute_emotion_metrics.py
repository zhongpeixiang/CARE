import argparse
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


def read_file(path):
    with open(path, "r") as f:
        data = [int(line.strip("\n")) for line in f.readlines()]
    return np.array(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_emotions_path', type=str, required=True)
    parser.add_argument('--target_emotions_path', type=str, required=True)

    args = parser.parse_args()
    output_emotions_path = args.output_emotions_path
    target_emotions_path = args.target_emotions_path
    
    output_emotions = read_file(output_emotions_path)
    target_emotions = read_file(target_emotions_path)
    
    accuracy = np.sum(output_emotions == target_emotions)/len(output_emotions)
    print("Emotion generation accuracy: {0}".format(accuracy))
    print("Classification report: \n")
    print(classification_report(target_emotions, output_emotions))