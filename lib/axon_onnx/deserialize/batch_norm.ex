defmodule AxonOnnx.Deserialize.BatchNorm do
  import AxonOnnx.Deserialize.Shared
  alias Onnx.NodeProto, as: Node

  require Logger

  def decode_node(axon, node, _opts) do
    %Node{
      input: [inp, gamma, beta, mean, var],
      output: [output_name],
      attribute: attrs
    } = node
    options = options!(attrs)

    mode = options["training_mode"] || 0
    epsilon = options["epsilon"] || 1.0e-5
    momentum = options["momenutm"] || 0.9

    if mode == 1 do
      Logger.warn("Training mode in batch norm has no effect")
    end

    inp = layer!(axon, inp)

    gamma = param!(gamma, params)
    beta = param!(beta, params)
    mean = param!(mean, params)
    var = param!(var, params)

    layer = Axon.batch_norm(inp, name: output_name, momentum: momentum, epsilon: epsilon)
    {layer, Map.put(used_params, output_name, %{
      "gamma" => gamma,
      "beta" => beta,
      "mean" => mean,
      "var" => var
    })}
  end
end

