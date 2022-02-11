defmodule AxonOnnx.Deserialize.If do
  import AxonOnnx.Deserialize.Shared
  alias Onnx.NodeProto, as: Node

  # Builds a conditional `If` layer.
  def decode_node(axon, node, _opts) do
    %Node{input: [input], attribute: attrs, output: outputs} = node
    cond_options = options!(attrs)
    inp = layer!(axon, input)

    else_branch = cond_options["else_branch"]
    then_branch = cond_options["then_branch"]

    # TODO: Don't match
    {[else_graph], else_params} = graph_to_axon(else_branch, [])
    {[then_graph], then_params} = graph_to_axon(then_branch, [])

    layers = Enum.reduce(outputs, [], fn out_name, acc ->
      Axon.cond(inp, & &1, then_graph, else_graph) ++ acc
    end)

    updated_params =
      else_params
      |> Map.merge(then_params)
      |> Map.merge(used_params)

    {layers, updated_params}
  end
end
