defmodule AxonOnnx.Deserialize.Reduction do
  import AxonOnnx.Deserialize.Shared
  alias Onnx.NodeProto, as: Node

  @op_decoder %{
    "ArgMax" => {&Nx.argmax/2, :axis},
    "ArgMin" => {&Nx.argmin/2, :axis},
    "ReduceMax" => {&Nx.reduce_max/2, :axes},
    "ReduceMean" => {&Nx.mean/2, :axes},
    "ReduceMin" => {&Nx.reduce_min/2, :axes},
    "ReduceProd" => {&Nx.product/2, :axes},
    "ReduceLogSum" => {&Nx.log(Nx.sum(&1, &2)), :axes}, # TODO: I think there is a stability problem here
    "ReduceSumSquare" => {&Nx.sum(Nx.power(&1, 2), &2), :axes},

    "ReduceLogSumExp" => {fn x, opts ->
       x
       |> Nx.exp()
       |> Nx.sum(opts)
       |> Nx.log()
     end, :axes}
  }

  # Builds a generic reduction layer by applying the given reduction operation
  # to the input in a custom layer.
  def decode_node(axon, node, _opts) do
    %Node{input: [input], attribute: attrs, output: [output_name], op_type: op_type} = node

    reduce_options = options!(attrs)

    %Axon{output_shape: shape} = axon_input = layer!(axon, input)

    keepdims = reduce_options["keepdims"] || 1
    keep_axes = if keepdims == 1, do: true, else: false

    {op, dim} = if Map.has_key?(@op_decoder, op_type) do
      Map.get(@op_decoder, op_type)
    else
      raise ArgumentError, "reduction #{op_type} is not supported"
    end

    axes =
      if dim == :axis do
        axis = reduce_options["axis"] || 0
        Nx.Shape.normalize_axis(shape, axis, List.duplicate(nil, Nx.rank(shape) - 1))
      else
        axes = reduce_options["axes"] || Nx.axes(shape)
        Nx.Shape.normalize_axes(shape, axes, List.duplicate(nil, Nx.rank(shape) - 1))
      end

    opts =
      if dim == :axis do
        last_index = reduce_options["select_last_index"] || 0
        tie_break = if last_index == 0, do: :low, else: :high
        [keep_axis: keep_axes, axis: axes, tie_break: tie_break]
      else
        [keep_axes: keep_axes, axes: axes]
      end

    out_shape =
      if keep_axes do
        Enum.reduce(List.wrap(axes), shape, fn x, shape -> put_elem(shape, x, 1) end)
      else
        shape = for i <- Nx.axes(shape), i not in List.wrap(axes), do: i
        List.to_tuple(shape)
      end

    Axon.layer(axon_input, reduce_fun, out_shape, %{}, output_name, opts)
  end
end
