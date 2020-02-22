import tensorflow as tf

ignore = {
    "Placeholder",
    "Const",
    "Identity",
}

with open("mobilenet_v1_1.0_224_frozen.pb", "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

    ops = []
    for node in graph_def.node:
        ops.append(node.op)

    for op in dict.fromkeys(ops):
        if op not in ignore:
            print(op)
