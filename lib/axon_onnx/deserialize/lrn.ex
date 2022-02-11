defmodule AxonOnnx.Deserialize.LRN do
  import AxonOnnx.Deserialize.Shared
  alias Onnx.NodeProto, as: Node

  def decode_node(axon, node, _opts) do
    %Node{input: [input], attribute: attrs, output: [output_name]} = node
    inp = layer!(axon, input)
    lrn_options = options!(attrs)

    alpha = lrn_options["alpha"] || 0.0001
    beta = lrn_options["beta"] || 0.75
    bias = lrn_options["bias"] || 1.0
    size = lrn_options["size"]

    axes = Enum.to_list(0..(size - 1))

    fun = fn x ->
      squares = Nx.power(x, 2)
      sum_squares = Nx.sum(squares, axes: axes, keep_axes: true)
      denom = Nx.power(Nx.add(bias, Nx.divide(alpha, Nx.multiply(size, sum_squares))), beta)
      Nx.divide(x, denom)
    end

    Axon.nx(inp, fun, name: output_name)
  end
end
