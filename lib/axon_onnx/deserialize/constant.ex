defmodule AxonOnnx.Deserialize.Constant do
  alias AxonOnnx.Deserialize.Shared
  alias Onnx.NodeProto, as: Node

  # Builds an Axon layer which returns a constant with the given value.
  # Constants are embedded in custom layers which just yield the value of the
  # constant here. They are not treated as parameters
  def decode_node(_axon, node, _opts) do
    %Node{attribute: attrs, output: [output_name]} = node
    constant_options = Shared.options!(attrs)

    cond do
      constant_options["sparse_value"] ->
        raise ArgumentError, "sparse tensors are not supported"

      constant_options["value"] ->
        Axon.constant(Shared.tensor!(constant_options["value"]), namme: output_name)

      constant_options["value_float"] ->
        Axon.constant(Nx.tensor(constant_options["value_float"], type: {:f, 32}),
          name: output_name
        )

      constant_options["value_floats"] ->
        Axon.constant(Nx.tensor(constant_options["value_floats"], type: {:f, 32}),
          name: output_name
        )

      constant_options["value_int"] ->
        Axon.constant(Nx.tensor(constant_options["value_int"], type: {:s, 64}),
          name: output_name
        )

      constant_options["value_ints"] ->
        Axon.constant(Nx.tensor(constant_options["value_ints"], type: {:s, 64}),
          name: output_name
        )

      constant_options["value_string"] or constant_options["value_strings"] ->
        raise ArgumentError, "string tensors are not supported"

      true ->
        raise ArgumentError, "invalid constant tensor type"
    end
  end
end
