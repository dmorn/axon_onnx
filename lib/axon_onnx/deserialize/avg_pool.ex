defmodule AxonOnnx.Deserialize.AvgPool do
  import AxonOnnx.Deserialize.Shared
  alias Onnx.NodeProto, as: Node

  def decode_node(axon, node, _opts) do
    %Node{input: [inp], attribute: attrs, output: [output_name]} = node
    avg_pool_options = options!(attrs)

    kernel_shape = avg_pool_options["kernel_shape"]
    ceil_mode = avg_pool_options["ceil_mode"] || 0
    auto_pad = avg_pool_options["auto_pad"] || "NOTSET"
    _count_include_pad = avg_pool_options["count_include_pad"] || 0
    pads = avg_pool_options["pads"]
    strides = avg_pool_options["strides"]
    dilations = avg_pool_options["dilations"]

    # Kernel size is a list of integers
    kernel_size = List.to_tuple(kernel_shape)

    # Axon only supports default ceil_mode right now
    if ceil_mode != 0 do
      raise ArgumentError,
            "invalid ceil_mode #{inspect(ceil_mode)}, Axon only supports" <>
              " ceil_mode of 0"
    end

    # Axon only supports count_include_pad == 1
    # if count_include_pad != 1 do
    #   raise ArgumentError, "invalid count_include_pad #{inspect(count_include_pad)}," <>
    #                           " Axon only supports mode 1"
    # end

    # Axon default strides are equal to the kernel shape (Keras behavior)
    # where as strides default to 1 in ONNX
    strides =
      if strides do
        strides
      else
        List.duplicate(1, tuple_size(kernel_size))
      end

    %Axon{output_shape: shape} = inp = layer!(axon, inp)

    # Compute padding from auto_pad and pads attributes
    padding_config = padding!(auto_pad, pads, shape, kernel_size, strides)

    Axon.avg_pool(inp,
      kernel_size: kernel_size,
      strides: strides,
      padding: padding_config,
      dilations: dilations,
      name: output_name
    )
  end
end
