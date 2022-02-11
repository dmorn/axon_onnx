defmodule AxonOnnx.Deserialize.Cast do
  import AxonOnnx.Deserialize.Shared
  alias Onnx.NodeProto, as: Node

  def decode_node(axon, node, _opts) do
    %Node{attribute: attrs, input: [input], output: [output_name]} = node
    cast_options = options!(attrs)
    inp = layer!(axon, input)

    fun = fn x ->
      case cast_options["to"] do
        1 ->
          Nx.as_type(x, {:f, 32})

        2 ->
          Nx.as_type(x, {:u, 8})

        3 ->
          Nx.as_type(x, {:s, 8})

        4 ->
          Nx.as_type(x, {:u, 16})

        5 ->
          Nx.as_type(x, {:s, 16})

        6 ->
          Nx.as_type(x, {:s, 32})

        7 ->
          Nx.as_type(x, {:s, 64})

        8 ->
          raise ArgumentError, "unsupported STRING type"

        9 ->
          raise ArgumentError, "unsupported BOOL type"

        10 ->
          Nx.as_type(x, {:f, 16})

        11 ->
          Nx.as_type(x, {:f, 64})

        12 ->
          Nx.as_type(x, {:u, 32})

        13 ->
          Nx.as_type(x, {:u, 64})

        14 ->
          raise ArgumentError, "unsupported COMPLEX type"

        15 ->
          raise ArgumentError, "unsupported COMPLEX type"

        16 ->
          Nx.as_type(x, {:bf, 16})
      end
    end

    Map.put(axon, output_name, Axon.nx(inp, fun, name: output_name))
  end
end
