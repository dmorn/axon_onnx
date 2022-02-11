defmodule AxonOnnx.Deserialize.MaxPool do
  import AxonOnnx.Deserialize.Shared
  alias Onnx.NodeProto, as: Node

  require Logger

  def decode_node(axon, node, _opts) do
    %Node{input: [inp], attribute: attrs, output: [output_name]} = node
    max_pool_options = options!(attrs)

    kernel_shape = max_pool_options["kernel_shape"]
    ceil_mode = max_pool_options["ceil_mode"] || 0
    auto_pad = max_pool_options["auto_pad"] || "NOTSET"
    storage_order = max_pool_options["storage_order"]
    pads = max_pool_options["pads"]
    strides = max_pool_options["strides"]
    dilations = max_pool_options["dilations"]

    # Kernel size is a list of integers
    kernel_size = List.to_tuple(kernel_shape)

    # Axon only supports default ceil_mode right now
    if ceil_mode != 0 do
      raise ArgumentError,
            "invalid ceil_mode #{inspect(ceil_mode)}, Axon only supports" <>
              " ceil_mode of 0"
    end

    # Storage Order is not an Axon concern
    if storage_order do
      Logger.warning(
        "Storage order is not supported by Axon and is instead a backend-specific" <>
          " detail. Your model might behave differently from the imported version if" <>
          " the storage order differs"
      )
    end

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

    Axon.max_pool(inp,
      kernel_size: kernel_size,
      strides: strides,
      padding: padding_config,
      dilations: dilations,
      name: output_name
    )
  end
end
