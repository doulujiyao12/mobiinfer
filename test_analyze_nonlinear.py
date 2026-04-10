import json
# 测试代码
with open('transformers/llm/export/model-2B/llm.mnn.json', 'r') as f:
    data = json.load(f)

binary_ops = set()
unary_ops = set()
layernorms = set()

for op in data.get('oplists', []):
    op_type = op.get('type')
    main = op.get('main', {})
    if op_type == 'BinaryOp':
        binary_ops.add(main.get('opType'))
    elif op_type == 'UnaryOp':
        unary_ops.add(main.get('opType'))
    elif op_type == 'LayerNorm':
        axis_size = len(main.get('axis', []))
        epsilon = main.get('epsilon')
        layernorms.add((axis_size, epsilon))

print("BinaryOp opTypes:", binary_ops)
print("UnaryOp opTypes:", unary_ops)
print("LayerNorm properties (len(axis), epsilon):", layernorms)
