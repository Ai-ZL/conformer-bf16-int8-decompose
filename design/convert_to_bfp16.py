import onnx
import numpy as np




onnx_model = onnx.load('model_I8I8_qops_static_symw+a2_4.onnx ')

graph = onnx_model.graph
node = graph.node


all_initializer = onnx_model.graph.initializer
all_value_info = onnx_model.graph.value_info

try:
    onnx.checker.check_model(onnx_model)
except onnx.checker.ValidationError as e:
    print('The model is invalid: %s' % e)
else:
    print('The model is valid!')


print("original version ",onnx_model.opset_import)

mode='bf16'

if mode =='bf16':#convert model to full bfloat16
    ##################################change model initializer###################################
    for i, j in enumerate(all_initializer):
        # change sigmoid initializer (fp32 -> bf16)
        if j.name[-21:] == 'Sigmoid_Exp_ln2_value':
            print("######%s######" % i)
            old_ini = j
            print('======change initiator: ', old_ini.name, '=========')
            graph.initializer.remove(old_ini)
            initializer_new = onnx.helper.make_tensor(

                name=old_ini.name,

                data_type=onnx.helper.TensorProto.DataType.BFLOAT16,
                vals=[np.log(2)],
                dims=old_ini.dims
            )
            graph.initializer.insert(i, initializer_new)
        elif j.name[-22:] == 'Sigmoid_Exp_base_value':
            print("######%s######" % i)
            old_ini = j
            print('======change initiator: ', old_ini.name, '=========')
            graph.initializer.remove(old_ini)
            initializer_new = onnx.helper.make_tensor(

                name=old_ini.name,

                data_type=onnx.helper.TensorProto.DataType.BFLOAT16,
                vals=[2],
                dims=old_ini.dims
            )
            graph.initializer.insert(i, initializer_new)
        elif j.name[-13:] == 'Sigmoid_add_b':
            print("######%s######" % i)
            old_ini = j
            print('======change initiator: ', old_ini.name, '=========')
            graph.initializer.remove(old_ini)
            initializer_new = onnx.helper.make_tensor(

                name=old_ini.name,

                data_type=onnx.helper.TensorProto.DataType.BFLOAT16,
                vals=old_ini.float_data,
                dims=old_ini.dims
            )
            graph.initializer.insert(i, initializer_new)
        elif j.name[-16:] == 'Sigmoid_divide_a':
            print("######%s######" % i)
            old_ini = j
            print('======change initiator: ', old_ini.name, '=========')
            graph.initializer.remove(old_ini)
            initializer_new = onnx.helper.make_tensor(

                name=old_ini.name,

                data_type=onnx.helper.TensorProto.DataType.BFLOAT16,
                vals=old_ini.float_data,
                dims=old_ini.dims
            )
            graph.initializer.insert(i, initializer_new)
        # change softmax initializer (fp32 -> bf16)
        elif j.name[-21:] == 'Softmax_Exp_ln2_value':
            print("######%s######" % i)
            old_ini = j
            print('======change initiator: ', old_ini.name, '=========')
            graph.initializer.remove(old_ini)
            initializer_new = onnx.helper.make_tensor(

                name=old_ini.name,

                data_type=onnx.helper.TensorProto.DataType.BFLOAT16,
                vals=[np.log(2)],
                dims=old_ini.dims
            )
            graph.initializer.insert(i, initializer_new)
        elif j.name[-22:] == 'Softmax_Exp_base_value':
            print("######%s######" % i)
            old_ini = j
            print('======change initiator: ', old_ini.name, '=========')
            graph.initializer.remove(old_ini)
            initializer_new = onnx.helper.make_tensor(

                name=old_ini.name,

                data_type=onnx.helper.TensorProto.DataType.BFLOAT16,
                vals=[2],
                dims=old_ini.dims
            )
            graph.initializer.insert(i, initializer_new)

###############################change model node###################################
    for i, j in enumerate(node):
        # add bf16 cast before sigmoid
        if j.name[-11:]=='Sigmoid_Neg':
            print("######%s######" % i)
            after_node = j
            print('======add bf16 cast before ', after_node.name, '=========')
            cast_name = after_node.name + "_Cast"
            cast_output = after_node.output[0] + "_cast"
            new_scale_node = onnx.helper.make_node(
                op_type="Cast",
                inputs=[after_node.input[0]],
                outputs=[cast_output],
                name=cast_name,
                to=getattr(onnx.TensorProto, "BFLOAT16")
            )
            graph.node.insert(i + 1, new_scale_node)
            after_node.input[0] = cast_output
        # add fp32 cast after sigmoid
        elif j.name[-31:]=='Sigmoid_output_0_QuantizeLinear':
            print("######%s######" % i)
            after_node = j
            print('======add fp32 cast before ', after_node.name, '=========')
            cast_name = after_node.name + "_Cast"
            cast_output = after_node.output[0] + "_cast"
            new_scale_node = onnx.helper.make_node(
                op_type="Cast",
                inputs=[after_node.input[0]],
                outputs=[cast_output],
                name=cast_name,
                to=getattr(onnx.TensorProto, "FLOAT")
            )
            graph.node.insert(i + 1, new_scale_node)
            after_node.input[0] = cast_output
        # add cast(to bf16) before softmax
        elif j.name[-15:]=='self_attn/Where':
            print("######%s######" % i)
            before_node = j
            print('======add bf16 cast after ', before_node.name, '=========')
            cast_name = before_node.name + "_Cast"
            cast_output = before_node.output[0] + "_cast"
            new_scale_node = onnx.helper.make_node(
                op_type="Cast",
                inputs=[before_node.output[0]],
                outputs=[cast_output],
                name=cast_name,
                to=getattr(onnx.TensorProto, "BFLOAT16")
            )
            graph.node.insert(i + 1, new_scale_node)
        elif j.name[-27:]=='self_attn/Softmax_ReduceMax':
            after_node = j
            print('======add bf16 cast before ', after_node.name, '=========')
            after_node.input[0] = after_node.input[0] + "_cast"
        elif j.name[-21:]=='self_attn/Softmax_Sub':
            after_node = j
            print('======add bf16 cast before ', after_node.name, '=========')
            after_node.input[0] = after_node.input[0] + "_cast"
        elif j.name=='/Transpose_3_output_0_DequantizeLinear':
            print("######%s######" % i)
            before_node = j
            print('======add bf16 cast after ', before_node.name, '=========')
            cast_name = before_node.name + "_Cast"
            cast_output = before_node.output[0] + "_cast"
            new_scale_node = onnx.helper.make_node(
                op_type="Cast",
                inputs=[before_node.output[0]],
                outputs=[cast_output],
                name=cast_name,
                to=getattr(onnx.TensorProto, "BFLOAT16")
            )
            graph.node.insert(i + 1, new_scale_node)
        elif j.name=='/LogSoftmax_ReduceMax':
            after_node = j
            print('======add bf16 cast before ', after_node.name, '=========')
            after_node.input[0] = after_node.input[0] + "_cast"
        elif j.name=='/LogSoftmax_Sub':
            after_node = j
            print('======add bf16 cast before ', after_node.name, '=========')
            after_node.input[0] = after_node.input[0] + "_cast"
        # add cast(to fp32) after softmax
        elif j.name[-17:]=='self_attn/Where_1':
            print("######%s######" % i)
            after_node = j
            print('======add fp32 cast before ', after_node.name, '=========')
            cast_name = after_node.name + "_Cast"
            cast_output = after_node.output[0] + "_cast"
            new_scale_node = onnx.helper.make_node(
                op_type="Cast",
                inputs=[after_node.input[2]],
                outputs=[cast_output],
                name=cast_name,
                to=getattr(onnx.TensorProto, "FLOAT")
            )
            graph.node.insert(i + 1, new_scale_node)
            after_node.input[2] = cast_output
        elif j.name=='/LogSoftmax_Log':
            print("######%s######" % i)
            before_node = j
            print('======add fp32 cast after ', before_node.name, '=========')
            cast_name = before_node.name + "_Cast"
            cast_input = before_node.output[0] + "_cast"
            new_scale_node = onnx.helper.make_node(
                op_type="Cast",
                inputs=[cast_input],
                outputs=before_node.output,
                name=cast_name,
                to=getattr(onnx.TensorProto, "FLOAT")
            )
            graph.node.insert(i + 1, new_scale_node)
            before_node.output[0] = cast_input


################################change model output (value_info)#################################
    for value_id, value in enumerate(all_value_info):
        # change sigmoid output (fp32 -> bf16)
        if value.name[-16:] == 'Sigmoid_output_0':
            print('======change ',value.name, ' datatype=========')
            value.type.tensor_type.elem_type = 16
        # change softmax output (fp32 -> bf16)
        elif value.name[-26:] == 'Softmax_output_0_reducemax':
            print('======change ', value.name, ' datatype=========')
            value.type.tensor_type.elem_type = 16
        elif value.name == 'logprobs_reducemax':
            print('======change ', value.name, ' datatype=========')
            value.type.tensor_type.elem_type = 16
        elif value.name[-20:] == 'Softmax_output_0_sub':
            print('======change ', value.name, ' datatype=========')
            value.type.tensor_type.elem_type = 16
        elif value.name == 'logprobs_sub':
            print('======change ', value.name, ' datatype=========')
            value.type.tensor_type.elem_type = 16
        elif value.name[-20:] == 'Softmax_output_0_exp':
            print('======change ', value.name, ' datatype=========')
            value.type.tensor_type.elem_type = 16
        elif value.name[-26:] == 'Softmax_output_0_reducesum':
            print('======change ', value.name, ' datatype=========')
            value.type.tensor_type.elem_type = 16
        elif value.name[-16:] == 'Softmax_output_0':
            print('======change ', value.name, ' datatype=========')
            value.type.tensor_type.elem_type = 16
        elif value.name == 'logprobs':
            print('======change ', value.name, ' datatype=========')
            value.type.tensor_type.elem_type = 16

if mode =='bf_fp_16': #convert model part is float16, part is bfloat16 for inference
    ##################################change model initializer###################################
    for i, j in enumerate(all_initializer):
        # change sigmoid initializer (fp32 -> bf16)
        if j.name[-21:] == 'Sigmoid_Exp_ln2_value':
            print("######%s######" % i)
            old_ini = j
            print('======change initiator: ', old_ini.name, '=========')
            graph.initializer.remove(old_ini)
            initializer_new = onnx.helper.make_tensor(

                name=old_ini.name,

                data_type=onnx.helper.TensorProto.DataType.BFLOAT16,
                vals=[np.log(2)],
                dims=old_ini.dims
            )
            graph.initializer.insert(i, initializer_new)
        elif j.name[-22:] == 'Sigmoid_Exp_base_value':
            print("######%s######" % i)
            old_ini = j
            print('======change initiator: ', old_ini.name, '=========')
            graph.initializer.remove(old_ini)
            initializer_new = onnx.helper.make_tensor(

                name=old_ini.name,

                data_type=onnx.helper.TensorProto.DataType.FLOAT16,
                vals=[2],
                dims=old_ini.dims
            )
            graph.initializer.insert(i, initializer_new)
        elif j.name[-13:] == 'Sigmoid_add_b':
            print("######%s######" % i)
            old_ini = j
            print('======change initiator: ', old_ini.name, '=========')
            graph.initializer.remove(old_ini)
            initializer_new = onnx.helper.make_tensor(

                name=old_ini.name,

                data_type=onnx.helper.TensorProto.DataType.BFLOAT16,
                vals=old_ini.float_data,
                dims=old_ini.dims
            )
            graph.initializer.insert(i, initializer_new)
        elif j.name[-16:] == 'Sigmoid_divide_a':
            print("######%s######" % i)
            old_ini = j
            print('======change initiator: ', old_ini.name, '=========')
            graph.initializer.remove(old_ini)
            initializer_new = onnx.helper.make_tensor(

                name=old_ini.name,

                data_type=onnx.helper.TensorProto.DataType.BFLOAT16,
                vals=old_ini.float_data,
                dims=old_ini.dims
            )
            graph.initializer.insert(i, initializer_new)
        # change softmax initializer (fp32 -> bf16)
        elif j.name[-21:] == 'Softmax_Exp_ln2_value':
            print("######%s######" % i)
            old_ini = j
            print('======change initiator: ', old_ini.name, '=========')
            graph.initializer.remove(old_ini)
            initializer_new = onnx.helper.make_tensor(

                name=old_ini.name,

                data_type=onnx.helper.TensorProto.DataType.BFLOAT16,
                vals=[np.log(2)],
                dims=old_ini.dims
            )
            graph.initializer.insert(i, initializer_new)
        elif j.name[-22:] == 'Softmax_Exp_base_value':
            print("######%s######" % i)
            old_ini = j
            print('======change initiator: ', old_ini.name, '=========')
            graph.initializer.remove(old_ini)
            initializer_new = onnx.helper.make_tensor(

                name=old_ini.name,

                data_type=onnx.helper.TensorProto.DataType.FLOAT16,
                vals=[2],
                dims=old_ini.dims
            )
            graph.initializer.insert(i, initializer_new)

    ###############################change model node###################################
    for i, j in enumerate(node):
        # add bf16 cast before sigmoid
        if j.name[-11:] == 'Sigmoid_Neg':
            print("######%s######" % i)
            after_node = j
            print('======add bf16 cast before ', after_node.name, '=========')
            cast_name = after_node.name + "_Cast"
            cast_output = after_node.output[0] + "_cast"
            new_scale_node = onnx.helper.make_node(
                op_type="Cast",
                inputs=[after_node.input[0]],
                outputs=[cast_output],
                name=cast_name,
                to=getattr(onnx.TensorProto, "BFLOAT16")
            )
            graph.node.insert(i + 1, new_scale_node)
            after_node.input[0] = cast_output
        # add fp32 cast after sigmoid
        elif j.name[-31:] == 'Sigmoid_output_0_QuantizeLinear':
            print("######%s######" % i)
            after_node = j
            print('======add fp32 cast before ', after_node.name, '=========')
            cast_name = after_node.name + "_Cast"
            cast_output = after_node.output[0] + "_cast"
            new_scale_node = onnx.helper.make_node(
                op_type="Cast",
                inputs=[after_node.input[0]],
                outputs=[cast_output],
                name=cast_name,
                to=getattr(onnx.TensorProto, "FLOAT")
            )
            graph.node.insert(i + 1, new_scale_node)
            after_node.input[0] = cast_output
        # add cast(to bf16) before softmax
        elif j.name[-15:] == 'self_attn/Where':
            print("######%s######" % i)
            before_node = j
            print('======add bf16 cast after ', before_node.name, '=========')
            cast_name = before_node.name + "_Cast"
            cast_output = before_node.output[0] + "_cast"
            new_scale_node = onnx.helper.make_node(
                op_type="Cast",
                inputs=[before_node.output[0]],
                outputs=[cast_output],
                name=cast_name,
                to=getattr(onnx.TensorProto, "BFLOAT16")
            )
            graph.node.insert(i + 1, new_scale_node)
        elif j.name[-27:] == 'self_attn/Softmax_ReduceMax':
            after_node = j
            print('======add bf16 cast before ', after_node.name, '=========')
            after_node.input[0] = after_node.input[0] + "_cast"
        elif j.name[-21:] == 'self_attn/Softmax_Sub':
            after_node = j
            print('======add bf16 cast before ', after_node.name, '=========')
            after_node.input[0] = after_node.input[0] + "_cast"
        elif j.name == '/Transpose_3_output_0_DequantizeLinear':
            print("######%s######" % i)
            before_node = j
            print('======add bf16 cast after ', before_node.name, '=========')
            cast_name = before_node.name + "_Cast"
            cast_output = before_node.output[0] + "_cast"
            new_scale_node = onnx.helper.make_node(
                op_type="Cast",
                inputs=[before_node.output[0]],
                outputs=[cast_output],
                name=cast_name,
                to=getattr(onnx.TensorProto, "BFLOAT16")
            )
            graph.node.insert(i + 1, new_scale_node)
        elif j.name == '/LogSoftmax_ReduceMax':
            after_node = j
            print('======add bf16 cast before ', after_node.name, '=========')
            after_node.input[0] = after_node.input[0] + "_cast"
        elif j.name == '/LogSoftmax_Sub':
            after_node = j
            print('======add bf16 cast before ', after_node.name, '=========')
            after_node.input[0] = after_node.input[0] + "_cast"
        # add cast(to fp32) after softmax
        elif j.name[-17:] == 'self_attn/Where_1':
            print("######%s######" % i)
            after_node = j
            print('======add fp32 cast before ', after_node.name, '=========')
            cast_name = after_node.name + "_Cast"
            cast_output = after_node.output[0] + "_cast"
            new_scale_node = onnx.helper.make_node(
                op_type="Cast",
                inputs=[after_node.input[2]],
                outputs=[cast_output],
                name=cast_name,
                to=getattr(onnx.TensorProto, "FLOAT")
            )
            graph.node.insert(i + 1, new_scale_node)
            after_node.input[2] = cast_output
        elif j.name == '/LogSoftmax_Log':
            print("######%s######" % i)
            before_node = j
            print('======add fp32 cast after ', before_node.name, '=========')
            cast_name = before_node.name + "_Cast"
            cast_input = before_node.output[0] + "_cast"
            new_scale_node = onnx.helper.make_node(
                op_type="Cast",
                inputs=[cast_input],
                outputs=before_node.output,
                name=cast_name,
                to=getattr(onnx.TensorProto, "FLOAT")
            )
            graph.node.insert(i + 1, new_scale_node)
            before_node.output[0] = cast_input

        # add fp16 cast before exp pow
        if j.name[-7:] == 'Exp_Pow':
            print("######%s######" % i)
            after_node = j
            print('======add fp16 cast before ', after_node.name, '=========')
            cast_name = after_node.name + "_Cast"
            cast_output = after_node.output[0] + "_cast"
            new_scale_node = onnx.helper.make_node(
                op_type="Cast",
                inputs=[after_node.input[1]],
                outputs=[cast_output],
                name=cast_name,
                to=getattr(onnx.TensorProto, "FLOAT16")
            )
            graph.node.insert(i + 1, new_scale_node)
            after_node.input[1] = cast_output
        # add bf16 cast after sigmoid exp pow
        elif j.name[-11:] == 'Sigmoid_Add':
            print("######%s######" % i)
            after_node = j
            print('======add bf16 cast before ', after_node.name, '=========')
            cast_name = after_node.name + "_Cast"
            cast_output = after_node.output[0] + "_cast"
            new_scale_node = onnx.helper.make_node(
                op_type="Cast",
                inputs=[after_node.input[0]],
                outputs=[cast_output],
                name=cast_name,
                to=getattr(onnx.TensorProto, "BFLOAT16")
            )
            graph.node.insert(i + 1, new_scale_node)
            after_node.input[0] = cast_output
        # add bf16 cast after softmax exp pow
        elif j.name[-17:] == 'Softmax_ReduceSum':
            print("######%s######" % i)
            after_node = j
            print('======add bf16 cast before ', after_node.name, '=========')
            cast_name = after_node.name + "_Cast"
            cast_output = after_node.output[0] + "_cast"
            new_scale_node = onnx.helper.make_node(
                op_type="Cast",
                inputs=[after_node.input[0]],
                outputs=[cast_output],
                name=cast_name,
                to=getattr(onnx.TensorProto, "BFLOAT16")
            )
            graph.node.insert(i + 1, new_scale_node)
            after_node.input[0] = cast_output
        elif j.name[-11:] == 'Softmax_Div':
            if j.name == '/LogSoftmax_Div':
                print("######%s######" % i)
                after_node = j
                print('======add bf16 cast before ', after_node.name, '=========')
                cast_output = after_node.output[0][:-4] + "_reducesum_cast"
                after_node.input[0] = cast_output
            else:
                print("######%s######" % i)
                after_node = j
                print('======add bf16 cast before ', after_node.name, '=========')
                cast_output = after_node.output[0] + "_reducesum_cast"
                after_node.input[0] = cast_output
        # add fp16 cast before softmax reducemax
        elif j.name[-17:] == 'Softmax_ReduceMax':
            print("######%s######" % i)
            after_node = j
            print('======add fp16 cast before ', after_node.name, '=========')
            cast_name = after_node.name + "_Cast"
            cast_output = after_node.output[0] + "_cast"
            new_scale_node = onnx.helper.make_node(
                op_type="Cast",
                inputs=[after_node.input[0]],
                outputs=[cast_output],
                name=cast_name,
                to=getattr(onnx.TensorProto, "FLOAT16")
            )
            graph.node.insert(i + 1, new_scale_node)
            after_node.input[0] = cast_output
        # add bf16 cast before softmax sub
        elif j.name[-11:] == 'Softmax_Sub':
            print("######%s######" % i)
            after_node = j
            print('======add bf16 cast before ', after_node.name, '=========')
            cast_name = after_node.name + "_Cast"
            cast_output = after_node.output[0] + "_cast"
            new_scale_node = onnx.helper.make_node(
                op_type="Cast",
                inputs=[after_node.input[1]],
                outputs=[cast_output],
                name=cast_name,
                to=getattr(onnx.TensorProto, "BFLOAT16")
            )
            graph.node.insert(i + 1, new_scale_node)
            after_node.input[1] = cast_output
        # add fp16 cast before softmax log
        elif j.name == '/LogSoftmax_Log':
            print("######%s######" % i)
            after_node = j
            print('======add fp16 cast before ', after_node.name, '=========')
            cast_name = after_node.name + "_Cast_2"
            cast_output = after_node.output[0] + "_cast"
            new_scale_node = onnx.helper.make_node(
                op_type="Cast",
                inputs=[after_node.input[0]],
                outputs=[cast_output],
                name=cast_name,
                to=getattr(onnx.TensorProto, "FLOAT16")
            )
            graph.node.insert(i + 1, new_scale_node)
            after_node.input[0] = cast_output

    ################################change model output (value_info)#################################
    for value_id, value in enumerate(all_value_info):
        # change sigmoid output (fp32 -> bf16)
        if value.name[-16:] == 'Sigmoid_output_0':
            print('======change ', value.name, ' datatype=========')
            value.type.tensor_type.elem_type = 16
        # change softmax output (fp32 -> bf16)
        elif value.name[-26:] == 'Softmax_output_0_reducemax':
            print('======change ', value.name, ' datatype=========')
            value.type.tensor_type.elem_type = 10
        elif value.name == 'logprobs_reducemax':
            print('======change ', value.name, ' datatype=========')
            value.type.tensor_type.elem_type = 10
        elif value.name[-20:] == 'Softmax_output_0_sub':
            print('======change ', value.name, ' datatype=========')
            value.type.tensor_type.elem_type = 16
        elif value.name == 'logprobs_sub':
            print('======change ', value.name, ' datatype=========')
            value.type.tensor_type.elem_type = 16
        elif value.name[-20:] == 'Softmax_output_0_exp':
            print('======change ', value.name, ' datatype=========')
            value.type.tensor_type.elem_type = 10
        elif value.name[-26:] == 'Softmax_output_0_reducesum':
            print('======change ', value.name, ' datatype=========')
            value.type.tensor_type.elem_type = 16
        elif value.name[-16:] == 'Softmax_output_0':
            print('======change ', value.name, ' datatype=========')
            value.type.tensor_type.elem_type = 16
        elif value.name == 'logprobs':
            print('======change ', value.name, ' datatype=========')
            value.type.tensor_type.elem_type = 16

onnx.save(onnx_model, 'model_final.onnx')