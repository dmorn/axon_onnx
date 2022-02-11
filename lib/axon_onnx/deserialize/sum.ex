defmodule AxonOnnx.Deserialize.Sum do
  import AxonOnnx.Deserialize.Shared
  alias Onnx.NodeProto, as: Node

  def decode_node(axon, node, _opts) do
    axons = for input <- inputs, do: layer!(axon, input)
    Axon.add(axons, name: output_name)
  end
end

