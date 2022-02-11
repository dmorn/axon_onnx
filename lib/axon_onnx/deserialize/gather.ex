defmodule AxonOnnx.Deserialize.Gather do
  import AxonOnnx.Deserialize.Shared
  alias Onnx.NodeProto, as: Node

  def decode_node(axon, node, [initializers: params]) do
    %Node{input: [x, ind], output: [output_name], attribute: attrs} = node
    gather_options = options!(attrs)

    axis = gather_options["axis"]

    {shape, input} = cond do
      Map.has_key?(params, x) ->
        tensor = params[x]
	{Nx.shape(tensor), tensor}
      Map.has_key?(axon, x) ->
        layer = axon[x]
	{layer.output_shape, layer}
    end

    inp_names = List.duplicate(nil, Nx.rank(shape))
    %Axon{output_shape: indices_shape} = indices = axon!(ind, axon)
    ind_names = List.duplicate(nil, Nx.rank(indices_shape))
    IO.inspect([shape, inp_names, indices_shape, ind_names, axis], label: "HERE")
    # output_shape = Nx.Shape.take(shape, inp_names, indices_shape, ind_names, axis)

    fun = fn x, indices ->
      Nx.take(x, Nx.as_type(indices, {:s, 64}), axis: axis)
    end

    Axon.layer([input, indices], fun, {}, %{}, output_name)
  end
end
