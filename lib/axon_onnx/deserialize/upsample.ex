defmodule AxonOnnx.Deserialize.Upsample do
  import AxonOnnx.Deserialize.Shared
  alias Onnx.NodeProto, as: Node

  # Builds an Axon layer which returns a new layer upsampling the input layer.
  # The scale activation layer must contain 1.0 as the first two values Each
  # dimension value of the output layer is:
  # `foutput_dimension = floor(input_dimension * scale)`
  def decode_node(axon, node, _opts) do
    node = %Node{attribute: attrs, input: [inp, scale], output: [output_name]}
    %Axon{output_shape: shape} = inp = layer!(axon, inp)
    %{"mode" => mode} = options!(attrs)
    scale = constant!(scale, axon, params)

    # Ignoring the first two 1.0 values to obtain the same dimension of scale_values 
    [_, _ | shape] = Tuple.to_list(shape)

    # Converting mode from string to atom to ensure Axon init and predict works correctly
    method =
      cond do
        is_binary(mode) -> String.to_atom(mode)
        is_atom(mode) -> mode
        true -> raise ArgumentError, "unsupported mode type. Must be string or atom, got: #{mode}"
      end

    output_shape =
      case Nx.to_flat_list(scale) do
        [1.0, 1.0 | scale_values] ->
          scale_values
          |> Enum.zip_with(shape, fn x, y -> floor(x * y) end)
          |> List.to_tuple()

        [s1, s2 | _] ->
          raise ArgumentError,
                "unspported scale format, first two scale values must be 1, got #{s1} and #{s2}"
      end

    Axon.resize(inp, output_shape, method: method, name: output_name)
  end
end
