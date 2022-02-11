defmodule AxonOnnx.Deserialize.Pad do
  import AxonOnnx.Deserialize.Shared
  alias Onnx.NodeProto, as: Node

  def decode_node(axon, node, _opts) do
    %Node{input: inputs, output: [output_name], attribute: attrs} = node
    pad_options = options!(attrs)

    case pad_options["mode"] do
      "constant" ->
        :ok

      nil ->
        :ok

      mode ->
        raise "unsupported padding mode #{inspect(mode)}"
    end

    [data, pads | maybe_constant] = inputs

    inp = layer!(axon, data)
    # TODO(seanmor5): Pads should probably be scrubbed from the graph
    # and parameters
    pads = param!(pads, params)

    padding_config =
      pads.ints
      |> Enum.chunk_every(2)
      |> Enum.zip()

    constant_value =
      case maybe_constant do
        [] ->
          0

        [value] ->
          tensor!(value)
      end

    Axon.pad(inp, padding_config, constant_value, name: output_name)
  end
end

