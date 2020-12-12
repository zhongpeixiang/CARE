import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    source_path = args.source
    output_path = args.output
    n = args.n

    with open(source_path, "r") as f:
        ctrl_output = [line.strip("\n").split(" ") for line in f.readlines() if line != "\n"]

    print(len(ctrl_output))
    outputs = []
    for line in ctrl_output:
        response_index = line.index("Response")
        outputs.append(" ".join(line[response_index + 2:]))
    
    with open(output_path, "w") as f:
        f.writelines([l+"\n" for l in outputs])

