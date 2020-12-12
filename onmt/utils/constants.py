emotion2id = {
    "neutral": 0,
    "joy": 1,
    "sadness": 2,
    "surprise": 3,
    "fear": 4,
    "anger": 5,
    "disgust": 6
}

id2emotion = {}
for e, id_ in emotion2id.items():
    id2emotion[id_] = e