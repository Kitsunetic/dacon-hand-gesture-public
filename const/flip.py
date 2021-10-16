from .label_names import id_to_label


bb = [
    *[[i, i + 10] for i in range(10)],
    *[[i, i + 25] for i in range(22, 45)],
    *[[i, i] for i in range(70, 99)],
    *[[i, i + 10] for i in range(100, 110)],
    *[[i, i + 25] for i in range(120, 145)],
    *[[i, i] for i in range(170, 196)],
]


id_flip = {}
for a, b in bb:
    if a in id_to_label and b in id_to_label:
        id_flip[id_to_label[a]] = id_to_label[b]
        id_flip[id_to_label[b]] = id_to_label[a]
