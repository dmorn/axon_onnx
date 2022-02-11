defmodule AxonOnnx.Deserialize.Reshape do
  import AxonOnnx.Deserialize.Shared
  alias Onnx.NodeProto, as: Node

  require Logger

  # TODO: This currently won't pass any Node tests because reshape
  # value is read in as an input, how do we handle that?
  def decode_node(axon, node, _opts) do
    %Node{input: [inp, shape], attribute: attrs, output: [output_name]} = node
    reshape_options = options!(attrs)
    allowzero = reshape_options["allowzero"] || 0

    inp = layer!(axon, inp)
    # Reshape is a constant value input that MUST be known
    # ahead of time so we can build a static graph, we can't
    # support any other reshape types
    shape = constant!(shape, axon, params)

    # We currently do not support zero sized dimensions
    if allowzero == 1 do
      Logger.warning(
        "Nx does not support zero-sized dimensions. If your reshape" <>
          " operation contains a zero-sized dimension, it will fail"
      )
    end

    new_shape =
      shape
      |> Nx.to_flat_list()
      |> List.to_tuple()

    Axon.reshape(inp, new_shape, name: output_name, ignore_batch?: false)
  end
end

