defmodule AxonOnnx.Deserialize.Dense do
  import AxonOnnx.Deserialize.Shared
  alias Onnx.NodeProto, as: Node

  # Builds an Axon dense layer from an ONNX MatMul or GEMM Node. MatMul
  # nodes do not account for bias (they're treated as a separate operation
  # in the graph). GEMM Nodes are a bit more in-depth.
  #
  # TODO(seanmor5): Handle alpha, beta attrs
  def decode_node(axon, node = %Node{op_type: "Gemm"}, _opts) do
    %Node{input: inputs, output: [output_name], attribute: attrs} = node
    [input, weight | maybe_bias] = inputs

    input = layer!(axon, input)
    weight = param!(weight, params)
    dense_options = options!(attrs)

    # TODO(seanmor5): Handle alpha, beta
    _alpha = dense_options["alpha"]
    _beta = dense_options["beta"]

    trans_a = dense_options["transA"]
    trans_b = dense_options["transB"]

    input =
      if trans_a == 1 do
        Nx.transpose(input)
      else
        input
      end

    weight =
      if trans_b == 1 do
        Nx.transpose(weight)
      else
        weight
      end

    {_, units} = Nx.shape(weight)

    layer = Axon.dense(input, units, use_bias: maybe_bias != [], name: output_name)

    updated_params =
      if maybe_bias == [] do
        Map.put(used_params, output_name, %{"kernel" => weight})
      else
        [bias] = maybe_bias
        bias = param!(bias, params)

        used_params
        |> Map.put(output_name, %{"kernel" => weight, "bias" => bias})
      end

    {layer, updated_params}
  end

  def decode_node(axon, node = %Node{op_type: "MatMul"}, _opts) do
    %Node{input: inputs, output: [output_name], attribute: attrs} = node
    [input, weight | maybe_bias] = inputs

    input = layer!(axon, input)
    weight = param!(weight, params)
    {_, units} = Nx.shape(weight)

    updated_params = Map.put(used_params, output_name, %{"kernel" => weight})
    layer = Axon.dense(input, units, use_bias: false, name: output_name)
    {layer, updated_params}
  end
end
