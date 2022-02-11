defmodule AxonOnnx.Deserialize.ElementWise do
  import AxonOnnx.Deserialize.Shared
  alias Onnx.NodeProto, as: Node

  @op_decoder %{
    "Abs" => &Nx.abs/1,
    "Acos" => &Nx.acos/1,
    "Acosh" => &Nx.acosh/1,
    "Asinh" => &Nx.asinh/1,
    "Atan" => &Nx.atan/1,
    "Atanh" => &Nx.atanh/1,
    "Ceil" => &Nx.ceil/1,
    "Cos" => &Nx.cos/1,
    "Cosh" => &Nx.cosh/1,
    "Erf" => &Nx.erf/1,
    "Floor" => &Nx.floor/1,
    "Log" => &Nx.log/1,
    "Not" => &Nx.not/1,
    "Round" => &Nx.round/1,
    "Sign" => &Nx.sign/1,
    "Sin" => &Nx.sin/1,
    "Sinh" => &Nx.sinh/1,
    "Sqrt" => &Nx.sqrt/1,
    "Tan" => &Nx.tan/1,

    "Identity" => & &1,
    "Neg" => &Nx.negate/1,

    "Shape" => fn x ->
      x
      |> Nx.shape()
      |> Tuple.to_list()
      |> Nx.tensor(backend: Nx.Defn.Expr)
    end,

    "Size" => fn x ->
      x
      |> Nx.size()
      |> Nx.tensor(backend: Nx.Defn.Expr)
    end,

    "HardSwish" => fn x ->
      alpha = Nx.divide(1, 6)
      beta = Nx.tensor(0.5)

      alpha
      |> Nx.multiply(x)
      |> Nx.add(beta)
      |> Nx.min(1)
      |> Nx.max(0)
      |> Nx.multiply(x)
    end,
  }

  # Builds a generic Nx layer by applying the given operation
  # to the input. Most of these functions are generic element-wise
  # operations such as Abs, Acos, etc.
  #
  # TODO(seanmor5): Replace with Axon.layer when we have better shape
  # inference
  def decode_node(axon, node, _opts) do
    %Node{input: [input_name], output: [output_name], op_type: op_type} = node
    input = layer!(axon, input_name)

    op = if Map.has_key?(@op_decoder, op_type) do
      Map.get(@op_decoder, op_type)
    else
      raise ArgumentError, "element-wise operation #{op_type} is not supported"
    end

    Axon.nx(input, op, name: output_name)
  end
end
