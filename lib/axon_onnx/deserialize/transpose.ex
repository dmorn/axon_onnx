defmodule AxonOnnx.Deserialize.Transpose do
  import AxonOnnx.Deserialize.Shared
  alias Onnx.NodeProto, as: Node

  # Builds an Axon transpose layer. Transpose is given by the perm option in
  # Node attribute. ONNX does not ignore batch dimensions, so that option is
  # always false.
  def decode_node(axon, node, _opts) do
    %Node{input: [input], attribute: attrs, output: [output_name]} = node
    transpose_options = options!(attrs)
    %Axon{output_shape: shape} = inp = layer!(axon, input)

    rank = Nx.rank(shape)
    permutation = transpose_options["perm"] || Enum.to_list((rank - 1)..0//-1)

    Axon.transpose(inp, permutation, name: output_name, ignore_batch?: false)
  end
end

