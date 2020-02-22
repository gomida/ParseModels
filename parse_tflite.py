from tflite import Model
from tflite.BuiltinOperator import BuiltinOperator 

data = open("./mobilenet_v1_1.0_224.tflite", "rb").read()
model = Model.Model.GetRootAsModel(data, 0)

# Function borrowed from https://github.com/apache/incubator-tvm
def build_str_map(obj):
    ret = {}
    for field_name in dir(obj):
        if not field_name.startswith('_'):
            field_value = getattr(obj, field_name)
            if isinstance(field_value, int):
                ret[field_value] = field_name
    return ret

builtinOps = build_str_map(BuiltinOperator)

for i in range(0, model.OperatorCodesLength()):
  print(builtinOps[model.OperatorCodes(i).BuiltinCode()])
