defmodule AxonOnnx.Deserialize.Conv do
  import AxonOnnx.Deserialize.Shared
  alias Onnx.NodeProto, as: Node

  def decode_node(axon, node, _opts) do
    %Node{attribute: attrs, input: input, output: [output_name]} = node
    conv_options = options!(attrs)

    kernel_shape = conv_options["kernel_shape"]
    auto_pad = conv_options["auto_pad"] || "NOTSET"
    dilations = conv_options["dilations"]
    group = conv_options["group"]
    pads = conv_options["pads"]
    strides = conv_options["strides"]

    # Kernel size is a list of integers
    kernel_size = List.to_tuple(kernel_shape)

    [inp, kernel | maybe_bias] = input

    %Axon{output_shape: shape} = axon_inp = layer!(axon, inp)

    padding_config = padding!(auto_pad, pads, shape, kernel_size, strides)

    kernel = param!(kernel, params)

    {units, kernel_size} =
      if kernel_shape do
        full_shape = Nx.shape(kernel)
        units = elem(full_shape, 0)

        shape =
          full_shape
          |> Tuple.delete_at(0)
          |> Tuple.delete_at(0)

        {units, shape}
      else
        full_shape = Nx.shape(kernel)
        units = elem(full_shape, 0)

        shape =
          full_shape
          |> Tuple.delete_at(0)
          |> Tuple.delete_at(0)

        {units, shape}
      end

    updated_params =
      if maybe_bias == [] do
        Map.put(used_params, output_name, %{"kernel" => kernel})
      else
        [bias] = maybe_bias
        bias = param!(bias, params)
        Map.put(used_params, output_name, %{"kernel" => kernel, "bias" => bias})
      end

    layer = Axon.conv(
      axon_inp,
      units,
      kernel_size: kernel_size,
      feature_group_size: group,
      kernel_dilation: dilations,
      padding: padding_config,
      strides: strides,
      use_bias: maybe_bias != [],
      name: output_name
    )
    {layer, updated_params}
  end
end
