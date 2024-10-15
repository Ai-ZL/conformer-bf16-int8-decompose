import onnx
import numpy as np
import argparse


#Arguments for running py files
parser = argparse.ArgumentParser()
parser.add_argument('-model', '--model_path', help='model stored path')
parser.add_argument('-model_o', '--model_path_out', help='output model stored path')
parser.add_argument('-m', '--mode', help='three decomposition mode: sigmoid, softmax, exp')
args = parser.parse_args()

onnx_model = onnx.load(args.model_path) #model need to modify
graph = onnx_model.graph # onnx model graph
node = graph.node # onnx model node


all_initializer = onnx_model.graph.initializer # onnx model initializer


#check model whether is valid
try:
    onnx.checker.check_model(onnx_model)
except onnx.checker.ValidationError as e:
    print('The model is invalid: %s' % e)
else:
    print('The model is valid!')

#check model opset version, all operator version should equal or older than this
print("original version ", onnx_model.opset_import)


############decompose all sigmoid function#################
if args.mode == 'sigmoid':
    for i,j in enumerate(node):
        if j.name[-7:] == 'Sigmoid': # The sigmoid node name should end with “Sigmoid”.
            print("######%s######" % i) # node index
            old_node = j
            onnx_model.graph.node.remove(j) # remove the old sigmoid node
            print('======update ',old_node.name,'=========')
            print("==========add initializer===========")
            initia_name = old_node.name+'_add_b'
            initia_name2 = old_node.name + '_divide_a'
            initializer_new = onnx.helper.make_tensor( #creat initializer

                name=initia_name,

                data_type=1,
                vals=[1],
                dims=[]
            )
            initializer_new2 = onnx.helper.make_tensor(

                name=initia_name2,

                data_type=1,
                vals=[1],
                dims=[]
            )
            graph.initializer.insert(0, initializer_new) # add new initializer
            graph.initializer.insert(0, initializer_new2)
            # 添加各种节点分解sigmoid (according to sigmoid function: y = 1 / (1 + exp(-x)))
            neg_output = old_node.output[0]+'_neg'
            neg_name = old_node.name+'_Neg'
            print("neg name is ",neg_name)
            print("neg output is ",neg_output)
            new_scale_node_1 = onnx.helper.make_node( # creat node
                op_type="Neg",
                inputs=old_node.input,
                outputs=[neg_output],
                name=neg_name,
            )
            exp_output = old_node.output[0]+'_exp'
            exp_name = old_node.name+'_Exp'
            new_scale_node_2 = onnx.helper.make_node(
                op_type="Exp",
                inputs=[neg_output],
                outputs=[exp_output],
                name=exp_name,
            )
            add_output = old_node.output[0]+'_add'
            add_name = old_node.name+'_Add'
            new_scale_node_3 = onnx.helper.make_node(
                op_type="Add",
                inputs=[exp_output, initia_name],
                outputs=[add_output],
                name=add_name,
            )
            div_name = old_node.name+'_Div'
            new_scale_node_4 = onnx.helper.make_node(
                op_type="Div",
                inputs=[initia_name2, add_output],
                outputs=old_node.output,
                name=div_name,
            )

            graph.node.insert(i, new_scale_node_1) # add new node
            graph.node.insert(i + 1, new_scale_node_2)
            graph.node.insert(i + 2, new_scale_node_3)
            graph.node.insert(i + 3, new_scale_node_4)




#############################decompose (Log)softmax function
if args.mode == 'softmax':
    select_axes = np.array([-1], dtype=np.int64)
    for i,j in enumerate(node):
        if j.name == "/LogSoftmax": # decompose logsoftmax node, log(softmax(x))
            print("######%s######" % i)
            old_node = j
            print('======update ', old_node.name, '=========')
            onnx_model.graph.node.remove(j) # remove logsoftmax node
            axes1 = onnx.numpy_helper.from_array(select_axes, "logprobs_reducesum_axes") #creat initializer
            # avoid overflows, subtract maximal value
            new_scale_node_2 = onnx.helper.make_node( # creat new node
                op_type="ReduceMax",
                inputs=old_node.input,
                outputs=['logprobs_reducemax'],
                name="/LogSoftmax_ReduceMax",
                axes=select_axes
            )
            new_scale_node_3 = onnx.helper.make_node(
                op_type="Sub",
                inputs=[old_node.input[0],
                        'logprobs_reducemax'],
                outputs=['logprobs_sub'],
                name="/LogSoftmax_Sub",
            )
            new_scale_node_1 = onnx.helper.make_node(
                op_type="Exp",
                inputs=['logprobs_sub'],
                outputs=['logprobs_exp'],
                name="/LogSoftmax_Exp",
            )
            new_scale_node_4 = onnx.helper.make_node(
                op_type="ReduceSum",
                inputs=['logprobs_exp','logprobs_reducesum_axes'],
                outputs=['logprobs_reducesum'],
                name="/LogSoftmax_ReduceSum",
            )
            new_scale_node_5 = onnx.helper.make_node(
                op_type="Div",
                inputs=['logprobs_exp', 'logprobs_reducesum'],
                outputs=['logprobs_div'],
                name="/LogSoftmax_Div",
            )
            new_scale_node_6 = onnx.helper.make_node(
                op_type="Log",
                inputs=['logprobs_div'],
                outputs=old_node.output,
                name="/LogSoftmax_Log",
            )
            graph.node.insert(i, new_scale_node_1) #insert new node
            all_initializer.insert(0, axes1) #insert new initializer
            graph.node.insert(i + 1, new_scale_node_2)
            graph.node.insert(i + 2, new_scale_node_3)
            graph.node.insert(i + 3, new_scale_node_4)
            graph.node.insert(i + 4, new_scale_node_5)
            graph.node.insert(i + 5, new_scale_node_6)

    select_axes2 = np.array([-1], dtype=np.int64)
    for i, j in enumerate(node): # decompose other softmax node
        if j.name[-8:] == "/Softmax":# The softmax node name should end with “/Softmax”.
            print("######%s######" % i)
            old_node = j
            print('======update ', old_node.name, '=========')
            onnx_model.graph.node.remove(j) #remove old sigmoid node
            initia_name = old_node.output[0]+"_reducesum_axes"
            axes1 = onnx.numpy_helper.from_array(select_axes2, initia_name) # creat new initializer
            # avoid overflows, subtract maximal value
            reducemax_name = old_node.name+"_ReduceMax"
            reducemax_output = old_node.output[0]+'_reducemax'
            new_scale_node_2 = onnx.helper.make_node( # creat new node
                op_type="ReduceMax",
                inputs=old_node.input,
                outputs=[reducemax_output],
                name=reducemax_name,
                axes=select_axes2
            )
            sub_output = old_node.output[0]+'_sub'
            sub_name = old_node.name+"_Sub"
            new_scale_node_3 = onnx.helper.make_node(
                op_type="Sub",
                inputs=[old_node.input[0],
                        reducemax_output],
                outputs=[sub_output],
                name=sub_name,
            )
            exp_output = old_node.output[0]+'_exp'
            exp_name = old_node.name+"_Exp"
            new_scale_node_1 = onnx.helper.make_node(
                op_type="Exp",
                inputs=[sub_output],
                outputs=[exp_output],
                name=exp_name,
            )
            reducesum_output = old_node.output[0]+'_reducesum'
            reducesum_name = old_node.name+"_ReduceSum"
            new_scale_node_4 = onnx.helper.make_node(
                op_type="ReduceSum",
                inputs=[exp_output, initia_name],
                outputs=[reducesum_output],
                name=reducesum_name,
            )
            div_name = old_node.name+"_Div"
            new_scale_node_5 = onnx.helper.make_node(
                op_type="Div",
                inputs=[exp_output, reducesum_output],
                outputs=old_node.output,
                name=div_name,
            )
            graph.node.insert(i, new_scale_node_1) # insert new node
            all_initializer.insert(0, axes1) #insert new initializer
            graph.node.insert(i + 1, new_scale_node_2)
            graph.node.insert(i + 2, new_scale_node_3)
            graph.node.insert(i + 3, new_scale_node_4)
            graph.node.insert(i + 4, new_scale_node_5)


#######################decompose exponential function##########
if args.mode=='exp':
    ln_2_value = np.array([np.log(2)], dtype=np.float32)
    base_value = np.array([2], dtype=np.float32)
    for i,j in enumerate(node):
        if j.name[-3:] == 'Exp':# The exp node name should end with “Exp”.
            print("######%s######" % i)
            old_node = j
            onnx_model.graph.node.remove(j) #remove the old exp node
            print('======update ', old_node.name, '=========')
            ln_2_name = old_node.name+"_ln2_value"
            value1 = onnx.numpy_helper.from_array(ln_2_value, ln_2_name) # add new initilizer
            base_name = old_node.name + "_base_value"
            value2 = onnx.numpy_helper.from_array(base_value, base_name)
            div_name = old_node.name + '_Div'
            div_output = old_node.output[0]+'_div'
            new_scale_node_1 = onnx.helper.make_node( # add new node
                op_type="Div",
                inputs=[old_node.input[0], ln_2_name],
                outputs=[div_output],
                name=div_name,
            )
            pow_name = old_node.name+'_Pow'
            new_scale_node_2 = onnx.helper.make_node(
                op_type="Pow",
                inputs=[base_name, div_output],
                outputs=old_node.output,
                name=pow_name,
            )
            all_initializer.insert(0, value1) # insert new initilizer
            all_initializer.insert(0, value2)
            graph.node.insert(i, new_scale_node_1) # insert new node
            graph.node.insert(i + 1, new_scale_node_2)


onnx.save(onnx_model, args.model_path_out) # save the modified model

