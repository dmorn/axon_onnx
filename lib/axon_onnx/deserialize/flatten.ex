defmodule AxonOnnx.Deserialize.Flatten do
  import AxonOnnx.Deserialize.Shared
  alias Onnx.NodeProto, as: Node

  def decode_node(axon, node, _opts) do
    %Node{input: [inp], output: [output_name]} = node
    inp = layer!(axon, inp)
    Axon.flatten(inp, name: output_name)
  end
end
