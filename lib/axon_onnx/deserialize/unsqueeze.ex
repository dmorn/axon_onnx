defmodule AxonOnnx.Deserialize.Unsqueeze do
  import AxonOnnx.Deserialize.Shared
  alias Onnx.NodeProto, as: Node

  # Builds an unsqueeze layer using a custom Nx layer with the given input and
  # axes.
  #
  # TODO(seanmor5): Use Axon.layer
  def decode_node(axon, node, _opts) do
    %Node{input: [input], attribute: attrs, output: [output_name]} = node
    unsqueeze_options = options!(attrs)

    inp = layer!(axon, input)
    axes = unsqueeze_options["axes"]

    fun = fn input ->
      Enum.reduce(axes, input, fn axis, x -> Nx.new_axis(x, axis) end)
    end

    case inp do
      %Nx.Tensor{} = tensor ->
        updated_params = Map.put(used_params, output_name, fun.(tensor))
        {[], updated_params}

      %Axon{} = model ->
        Axon.nx(model, fun, name: output_name)
    end
  end
end

