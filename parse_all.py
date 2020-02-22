import os
import tensorflow as tf
import xlsxwriter
from tflite import Model
from tflite.BuiltinOperator import BuiltinOperator

ignore = {
    # TF
    "Placeholder",
    "Const",
    "Identity",
    "Shape",
    "FIFOQueueV2",
    "QueueDequeueManyV2",
    "TensorArrayV3",
    "Enter",
    "Merge",
    "Range",
    "TensorArrayScatterV3",
    "TensorArraySizeV3",
    "TensorArrayReadV3",
    "TensorArrayWriteV3",
    "TensorArrayGatherV3",
    "NextIteration",
    "Less",
    "LoopCond",
    "Pack",
    "Unpack",
    "Switch",
    "Exit",
    "ExpandDims",
    "ZerosLike",
    "Tile",
    "Gather",
    "Assert",
    "Where",
    "TopKV2",
    "Fill",
    "Size",
    "NonMaxSuppressionV2",

    # TFLITE
    "DEQUANTIZE",
    "CUSTOM",
}

opMap = {
    "BiasAdd": "Add",
    "Add": "Add",
    "AddV2": "Add",
    "ADD": "Add",
    "RealDiv": "Div",
    "Equal": "Equal",
    "Exp": "Exp",
    "Greater": "Greater",
    "Mean": "Mean",
    "MEAN": "Mean",
    "Mul": "Mul",
    "MUL": "Mul",
    "Sub": "Sub",
    "SUB": "Sub",
    "Maximum": "Maximum",
    "Minimum": "Minimum",
    "AvgPool": "AvgPool",
    "AVERAGE_POOL_2D": "AvgPool",
    "MaxPool": "MaxPool",
    "MAX_POOL_2D": "MaxPool",
    "FusedBatchNorm": "BatchNorm",
    "FusedBatchNormV3": "BatchNorm",
    "ConcatV2": "Concat",
    "CONCATENATION": "Concat",
    "Cast": "Cast",
    "Conv2D": "Conv",
    "CONV_2D": "Conv",
    "DepthwiseConv2dNative": "GroupConv(Depthwise)",
    "DEPTHWISE_CONV_2D": "GroupConv(Depthwise)",
    "ResizeBilinear": "ResizeBilinear",
    "Relu": "Relu",
    "Relu6": "Relu6",
    "Sigmoid": "Sigmoid",
    "LOGISTIC": "Sigmoid",
    "Reshape": "Reshape",
    "RESHAPE": "Reshape",
    "Transpose": "Transpose",
    "Softmax": "Softmax",
    "SOFTMAX": "Softmax",
    "Squeeze": "Squeeze",
    "Pad": "Pad",
    "PAD": "Pad",
    "StridedSlice": "Slice",
    "Slice": "Slice",
    "Split": "Split",
}

opKey = list(dict.fromkeys(opMap.values()))

def parse_pb(path, worksheet, pos):
    f = open(path, "rb")
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

    ops = []
    for node in graph_def.node:
        ops.append(node.op)

    #print(path)
    worksheet.write(0, pos[0], os.path.basename(path))
    for op in dict.fromkeys(ops):
        if op not in ignore:
            #print(opMap[op])
            worksheet.write(opKey.index(opMap[op])+1, pos[0], "○")
    pos[0] += 1


def parse_tflite(path, worksheet, pos):
    f = open(path, "rb")
    model = Model.Model.GetRootAsModel(f.read(), 0)

    # Code borrowed from https://github.com/apache/incubator-tvm
    ops = {}
    for field_name in dir(BuiltinOperator):
        if not field_name.startswith('_'):
            field_value = getattr(BuiltinOperator, field_name)
            if isinstance(field_value, int):
                ops[field_value] = field_name

    #print(path)
    worksheet.write(0, pos[0], os.path.basename(path))
    for i in range(0, model.OperatorCodesLength()):
        opcode = model.OperatorCodes(i).BuiltinCode()
        op = ops[opcode]
        if op not in ignore:
            #print(opMap[op])
            worksheet.write(opKey.index(opMap[op])+1, pos[0], "○")
    pos[0] += 1


if __name__ == "__main__":
    workbook = xlsxwriter.Workbook('table.xlsx')
    worksheet = workbook.add_worksheet()
    pos = [0]
    for i, k in enumerate(opKey, start=1):
        worksheet.write(i, pos[0], k)
    pos[0] += 1

    dirname = "models"
    for fname in os.listdir(dirname):
        path = os.path.join(dirname, fname)
        _, ext = os.path.splitext(path)
        if ext == ".pb":
            parse_pb(path, worksheet, pos)
        elif ext == ".tflite":
            parse_tflite(path, worksheet, pos)

    workbook.close()
