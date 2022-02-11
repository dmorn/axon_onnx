defmodule AxonOnnx.Deserialize.BinElementWise do
  import AxonOnnx.Deserialize.Shared
  alias Onnx.NodeProto, as: Node

  @op_decoder %{
    "Add" => &Axon.add/2,
    "And" => &Nx.logical_and/2,
    "Div" => &Nx.divide/2,
    "Equal" => &Nx.equal/2,
    "Greater" => &Nx.greater/2,
    "GreaterOrEqual" => &Nx.greater_equal/2,
    "Less" => &Nx.less/2,
    "LessOrEqual" => &Nx.less_equal/2,
    "Mul" => &Axon.multiply/2,
    "Or" => &Nx.logical_or/2,
    "Pow" => &Nx.power/2,
    "Sub" => &Axon.subtract/2,
    "Xor" => &Nx.logical_xor/2,

    # TODO(seanmor5): Support fmod option
    "Mod" => fn {x, y} -> Nx.remainder(x, y) end,
  }

  # TODO: this one requires more options!
  def decode_node(axon, node = %Node{op_type: "BitShift"}, _opts) do
    raise ArgumentError, "bit-shift decoding is on hold"

    #  %Node{attribute: attrs} = op_node

    #  to_axon_binary_op(op_node, axon, params, used_params, fn x, y ->
    #    shift_options = options!(attrs)

    #    case shift_options["direction"] do
    #      "LEFT" ->
    #        Nx.left_shift(Nx.as_type(x, {:s, 64}), Nx.as_type(y, {:s, 64}))

    #      "RIGHT" ->
    #        Nx.right_shift(Nx.as_type(x, {:s, 64}), Nx.as_type(y, {:s, 64}))
    #    end
    #  end)
  end

  # Builds an Axon layer from an element-wise binary operation. Binary
  # op is either an atom representing one of Axon's legitimate Binary op
  # layers, or a function to be used in a custom layer.
  #
  # TODO(seanmor5): Verify broadcasting semantics
  def decode_node(axon, node, _opts) do
    %Node{input: [x, y], output: [output_name], op_type: op_type} = node
    %Axon{output_shape: s1} = inp1 = layer!(axon, x)
    %Axon{output_shape: s2} = inp2 = layer!(axon, y)

    # TODO: Must fix Axon.layer with no parameters
    out_shape = Axon.Shape.element_wise([s1, s2])

    op = if Map.has_key?(@op_decoder, op_type) do
      Map.get(@op_decoder, op_type)
    else
      raise ArgumentError, "binary element-wise operation #{op_type} is not supported"
    end

    Axon.layer([inp1, inp2], op, out_shape, %{}, name: output_name)
  end
end
