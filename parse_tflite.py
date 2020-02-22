from tflite import Model
from tflite.BuiltinOperator import BuiltinOperator 

with open("mobilenet_v1_1.0_224.tflite", "rb") as f:
    model = Model.Model.GetRootAsModel(f.read(), 0)

    # Code borrowed from https://github.com/apache/incubator-tvm
    ops = {}
    for field_name in dir(BuiltinOperator):
        if not field_name.startswith('_'):
            field_value = getattr(BuiltinOperator, field_name)
            if isinstance(field_value, int):
                ops[field_value] = field_name

    for i in range(0, model.OperatorCodesLength()):
        print(ops[model.OperatorCodes(i).BuiltinCode()])
