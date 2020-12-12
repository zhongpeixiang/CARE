from collections import Counter

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def filter_examples(data, label, num_classes, max_ex_per_class):
    new_data = []
    new_label=[]
    num_ex_per_class = [0]*num_classes
    for ex_data, ex_label in zip(data, label):
        if num_ex_per_class[ex_label] < max_ex_per_class:
            new_data.append(ex_data)
            new_label.append(ex_label)
            num_ex_per_class[ex_label] += 1
    return new_data, new_label

def remove_class(data, label, class_index):
    new_data = []
    new_label=[]
    for ex_data, ex_label in zip(data, label):
        if ex_label != class_index:
            new_data.append(ex_data)
            new_label.append(ex_label)
    # print(len(data), len(label), len(new_data), len(new_label))
    return new_data, new_label


def label_weights(label):
    label_counter = Counter(label)
    weight = []
    total = len(label)
    for label_idx in range(len(label_counter.keys())):
        weight.append(total/label_counter[label_idx])
    return weight