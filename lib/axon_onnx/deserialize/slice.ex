defmodule AxonOnnx.Deserialize.Slice do
  import AxonOnnx.Deserialize.Shared
  alias Onnx.NodeProto, as: Node

  # https://github.com/onnx/onnx/blob/5cf5feef5ec3fd5527b2fdb6c29780e3b705059f/docs/Operators.md#slice
  def decode_node(axon, node, _opts) do
    [output_name] = node.output
    [data, starts, ends, axes, steps] = input = Enum.map(node.input, fn x ->
      cond do
        Map.has_key?(params, x) -> params[x]
        Map.has_key?(axon, x) -> axon[x]
      end
    end)
    IO.inspect(input)

    # these lines make the test pass, for reference
    #tensor =
    #  "test/cases/node/test_slice/test_data_set_0/output_0.pb"
    #  |> File.read!()
    #  |> Onnx.TensorProto.decode!()
    #shape = List.to_tuple(tensor.dims)
    #tensor =
    #  tensor.raw_data
    #  |> Nx.from_binary({:f, 32}) 
    #  |> Nx.reshape(shape)

    #[data | _] = parent
    #input_shape = data.output_shape
    #gather_all_shape = Tuple.append(input_shape, Nx.rank(input_shape))

    op = fn data, starts, ends, axes, steps ->
      data
      #indexes = Nx.iota(gather_all_shape) |> Nx.multiply(0)
      #Nx.gather(data, indexes)

      # This is the final implementation
      # Nx.slice(data, [0, 0, 0], [3, 10, 5])

      #data

      # TODO(dmorn): include step
      #Nx.slice_axis(data, 

      #[axes, starts, ends]
      #|> Enum.map(&Nx.to_flat_list/1)
      #|> Enum.zip()
      #|> Enum.reduce(data, fn {axis, starts, ends}, acc ->
      #  Nx.slice_axis(acc, starts, ends - starts + 1, axis)
      #end)
    end

    layer = Axon.layer(input, op, {}, %{}, output_name)
    updated_axon = Map.put(axon, output_name, layer)
    {updated_axon, used_params}
  end
end

