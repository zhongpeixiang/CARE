import os
import argparse
import random


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--src', required=True, type=str,
                        help='Source translation file')
    parser.add_argument('--emotion_id', type=int, default=-1,
                        help='The emotion type to generate, default -1 indicates random from 0 to 5')

    args = parser.parse_args()
    src = args.src
    emotion_id = args.emotion_id
    # data_dir = os.path.dirname(src)
    # fname = os.path.basename(src)

    if emotion_id > 5:
        raise ValueError("emotion_id must be less than or equal to 5")
    
    with open(src, "r") as f:
        num_lines = len(f.readlines())
    
    # generate emotions
    if emotion_id < 0:
        emotions = [random.randint(0, 5) for i in range(num_lines)]
    else:
        emotions = [emotion_id]*num_lines
    
    # save emotions
    emotion_file = src.replace(".txt", "-input-emotions.txt")
    with open(emotion_file, "w") as f:
        f.writelines([str(emotion_id) + "\n" for emotion_id in emotions])