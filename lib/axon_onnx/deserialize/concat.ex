defmodule AxonOnnx.Deserialize.Concat do
  import AxonOnnx.Deserialize.Shared
  alias Onnx.NodeProto, as: Node

  # Builds an Axon layer which returns a new layer with input values 
  # concatenated on the given axis 
  def decode_node(axon, node, _opts) do
    %Node{attribute: attrs, input: inputs, output: [output_name]} = node
    inputs = for inp <- inputs, do: layer!(axon, inp)
    %{"axis" => axis} = options!(attrs)

    Axon.concatenate(inputs, axis: axis, name: output_name)
  end
end
