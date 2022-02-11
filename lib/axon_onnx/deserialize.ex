defmodule AxonOnnx.Deserialize do
  alias Onnx.ModelProto, as: Model
  alias Onnx.GraphProto, as: Graph
  alias Onnx.ValueInfoProto, as: Value
  alias Onnx.NodeProto, as: Node
  alias Onnx.TypeProto, as: Type
  alias Onnx.TensorProto, as: Tensor
  alias Onnx.TypeProto.Tensor, as: Placeholder
  alias Onnx.TensorShapeProto, as: Shape
  alias Onnx.TensorShapeProto.Dimension, as: Dimension

  alias AxonOnnx.Deserialize.ElementWise
  alias AxonOnnx.Deserialize.BinElementWise
  alias AxonOnnx.Deserialize.Reduction
  alias AxonOnnx.Deserialize.Activation
  alias AxonOnnx.Deserialize.Shared

  require Logger

  # TODO(seanmor5): Currently we do a lot of potentially expensive operations
  # eagerly (especially when manipulating parameters), we can potentially make
  # them part of the model or alternatively return an initialization function
  # which can be JIT-compiled.

  # TODO(seanmor5): The current approach builds a lot of intermediate graphs,
  # instead we should only keep graphs which are specified as outputs and override
  # all other graphs so they are GC'ed

  # TODO(seanmor5): Some operations occur strictly on parameters (e.g. reshape, unsqueeze,
  # etc.), so we need to change all of these cases to handle instances where the only
  # input is a parameter which is an Nx expression rather than a model

  # TODO(seanmor5): Because some operations act on parameter inputs which don't have a
  # parameterized equivalent operation in Axon (e.g. add, multiply, etc.), we need
  # a way to implement them that still builds an Axon model but preserves the parameters

  # TODO(seanmor5): Because there are multiple versions of the protocol, there are also
  # multiple versions of each function. It's not that unreasonable to try to support every
  # version, but it just makes for a lot of annoying edge cases. Standardize around a minimum
  # supported version for guaranteed compatibility

  @module_decoder %{
    "Abs" => ElementWise,
    "Acos" => ElementWise,
    "Acosh" => ElementWise,
    "Asin" => ElementWise,
    "Asinh" => ElementWise,
    "Atan" => ElementWise,
    "Atanh" => ElementWise,
    "Ceil" => ElementWise,
    "Cos" => ElementWise,
    "Cosh" => ElementWise,
    "Erf" => ElementWise,
    "Floor" => ElementWise,
    "Identity" => ElementWise,
    "Log" => ElementWise,
    "Neg" => ElementWise,
    "Not" => ElementWise,
    "Round" => ElementWise,
    "Sign" => ElementWise,
    "Sin" => ElementWise,
    "Sinh" => ElementWise,
    "Sqrt" => ElementWise,
    "Tan" => ElementWise,
    "Shape" => ElementWise,
    "Size" => ElementWise,
    "HardSwish" => ElementWise,

    "Add" => BinElementWise,
    "And" => BinElementWise,
    "Div" => BinElementWise,
    "Equal" => BinElementWise,
    "Greater" => BinElementWise,
    "GreaterOrEqual" => BinElementWise,
    "Less" => BinElementWise,
    "LessOrEqual" => BinElementWise,
    "Mod" => BinElementWise,
    "Mul" => BinElementWise,
    "Or" => BinElementWise,
    "Pow" => BinElementWise,
    "Sub" => BinElementWise,
    "Xor" => BinElementWise,
    "BitShift" => BinElementWise,

    "ArgMax" => Reduction,
    "ArgMin" => Reduction,
    "ReduceMax" => Reduction,
    "ReduceMean" => Reduction,
    "ReduceMin" => Reduction,
    "ReduceProd" => Reduction,
    "ReduceLogSum" => Reduction,
    "ReduceLogSumExp" => Reduction,
    "ReduceSumSquare" => Reduction,

    "Celu" => Activation,
    "Elu" => Activation,
    "Exp" => Activation,
    "HardSigmoid" => Activation,
    "LeakyRelu" => Activation,
    "LogSoftmax" => Activation,
    "Tanh" => Activation,
    "Relu" => Activation,
    "Selu" => Activation,
    "Sigmoid" => Activation,
    "Softmax" => Activation,
    "Softplus" => Activation,
    "Softsign" => Activation,

    "GlobalAveragePool" => GlobalPool,
    "GlobalLpPool" => GlobalPool,
    "GlobalMaxPool" => GlobalPool,

    "AveragePool" => AvgPool,
    "BatchNormalization" => BatchNorm,
    "Cast" => Cast,
    "Constant" => Constant,
    "Concat" => Concat,
    "Conv" => Conv,
    "Flatten" => Flatten,
    "Gemm" => Dense,
    "MatMul" => Dense,
    "Gather" => Gather,
    "Split" => Split,
    "Upsample" => Upsample,
    "If" => If,
    "LRN" => LRN,
    "Reshape" => Reshape,
    "Slice" => Slice,
    "Sum" => Sum,
    "Transpose" => Transpose,
    "Unsqueeze" => Unsqueeze,
    "MaxPool" => MaxPool,
    "Pad" => Pad,
  }

  def __import__(file, opts \\ []) do
    file
    |> File.read!()
    |> Model.decode!()
    |> to_axon(opts)
  end

  defp to_axon(%Model{graph: %Graph{} = graph}, opts) do
    dimensions = opts[:dimensions] || []

    {graph, params} = graph_to_axon(graph, dimensions)

    case graph do
      [graph] ->
        # single-output
        {graph, params}

      graph when is_list(graph) ->
        # multi-output
        {List.to_tuple(graph), params}
    end
  end

  def graph_to_axon(%Graph{node: nodes} = graph, dimensions) do
    params = get_params(graph)
    inputs = get_inputs(graph, params, dimensions)
    outputs = get_outputs(graph)
    {nodes, params} = get_nodes(nodes, inputs, params, %{})
    {Enum.map(outputs, fn name -> nodes[name] end), params}
  end

  defp get_inputs(%Graph{input: inputs}, params, dimensions) do
    Enum.reduce(inputs, %{}, fn x, acc ->
      %Value{name: name, type: %Type{value: value}} = x
      
      if Map.has_key?(params, name) do
        acc
      else
        case value do
          {:tensor_type, %Placeholder{} = tensor} ->
            input_shape = shape!(tensor, dimensions)
            Map.put(acc, name, Axon.input(input_shape))

          unsupported ->
            raise ArgumentError, "unsupported input type #{inspect(unsupported)}"
        end
      end
    end)
  end

  defp get_params(%Graph{initializer: initializer}) do
    Enum.reduce(initializer, %{}, fn %Tensor{name: name} = tensor, params ->
      Map.put(params, name, Shared.tensor!(tensor))
    end)
  end

  defp get_outputs(%Graph{output: outputs}) do
    Enum.map(outputs, fn %Value{name: name} -> name end)
  end
  
  defp get_nodes(nodes, inputs, params, used_params) do
    Enum.reduce(nodes, {inputs, used_params}, fn node, acc ->
      %Node{op_type: op_type} = node
      {axon, used_params} = acc

      # TODO: Why keeping track of parameters used?

      output =
        @module_decoder
        |> Map.get(op_type)
        |> apply(:decode_node, [node, [used_params: used_params, initializers: params]])

      case output do
        {layer, updated_params} ->
	  {Map.put(axon, layer.name, layer), updated_params}

	layers when is_list(layers) ->
          updated_axon =
            Enum.reduce(layers, axon, fn output, new_axon ->
              Map.put(new_axon, output.name, output)
            end)
	  {updated_axon, used_params}

	layer ->
	  {Map.put(axon, layer.name, layer), used_params}
      end
    end)
  end

  defp shape!(%Placeholder{shape: %Shape{dim: dims}}, dim_params) do
    dims
    |> Enum.map(fn %Dimension{value: value} ->
      case value do
        {:dim_value, val} ->
          val

        {:dim_param, key} ->
          unless Map.has_key?(dim_params, key) do
            raise "dimension #{inspect(key)} not found in provided dimensions," <>
                    " you must specify unknown dimension shapes at import time"
          end

          dim_params[key]

        _ ->
          raise ArgumentError, "unsupported dimension type"
      end
    end)
    |> List.to_tuple()
  end
end
