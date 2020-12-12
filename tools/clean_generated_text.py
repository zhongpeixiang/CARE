import argparse

def read_file(path):
    with open(path, "r") as f:
        return f.readlines()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, required=True)

    args = parser.parse_args()
    path = args.path
    
    outputs = read_file(path)
    
    cleaned_outputs = []
    for e in outputs:
        if e=="\n":
            e=".\n"
        cleaned_outputs.append(e)
    
    with open(path, "w") as f:
        f.writelines(cleaned_outputs)
