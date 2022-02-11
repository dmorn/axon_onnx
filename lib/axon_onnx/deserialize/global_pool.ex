defmodule AxonOnnx.Deserialize.GlobalPool do
  import AxonOnnx.Deserialize.Shared

  def decode_node(axon, node = %Node{op_type: "GlobalAveragePool"}, _opts) do
    %Node{input: [inp], output: [output_name]} = node
    inp = layer!(axon, inp)
    Axon.global_avg_pool(inp, name: output_name, keep_axes: true)
  end

  def decode_node(axon, node = %Node{op_type: "GlobalMaxPool"}, _opts) do
    %Node{input: [inp], output: [output_name]} = node
    inp = layer!(axon, inp)
    Axon.global_max_pool(inp, name: output_name, keep_axes: true)
  end

  def decode_node(axon, node = %Node{op_type: "GlobalLpPool"}, _opts) do
    %Node{attribute: attrs, input: [inp], output: [output_name]} = node
    inp = layer!(axon, inp)
    lp_pool_options = options!(attrs)
    Axon.global_lp_pool(inp, norm: lp_pool_options["p"], name: output_name)
  end
end
