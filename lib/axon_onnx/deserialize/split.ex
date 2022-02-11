defmodule AxonOnnx.Deserialize.Split do
  import AxonOnnx.Deserialize.Shared
  alias Onnx.NodeProto, as: Node

  def decode_node(axon, node, _opts) do
    %Node{attribute: attrs, input: [inp], output: output_names} = node
    inp = layer!(axon, inp)
    %{"axis" => axis, "split" => split_sizes} = options!(attrs)

    Axon.split(inp, split_sizes, axis: axis, name: output_names)
  end
end
