defmodule AxonOnnx.Deserialize.Activation do
  import AxonOnnx.Deserialize.Shared
  alias Onnx.NodeProto, as: Node

  @op_decoder %{
    "Celu" => {:celu, alpha: {"alpha", 1.0}},
    "Elu" => {:elu, alpha: {"alpha", 1.0}},
    "Exp" => {:exp, []},
    "HardSigmoid" => {:hard_sigmoid, alpha: {"alpha", 0.2}, beta: {"beta", 0.5}},
    "LeakyRelu" => {:leaky_relu, alpha: {"alpha", 0.01}},
    "LogSoftmax" => {:log_softmax, axis: {"axis", -1}},
    "Tanh" => {:tanh, []},
    "Relu" => {:relu, []},
    "Selu" => {:selu, alpha: {"alpha", 1.67326319217681884765625}, gamma: {"gamma", 1.05070102214813232421875}},
    "Sigmoid" => {:sigmoid, []},
    "Softmax" => {:softmax, axis: {"axis", -1}},
    "Softplus" => {:softplus, []},
    "Softsign" => {:softsign, []}
  }

  # Builds an axon activation layer with the given activation function.
  # `activation` must be a legitimate Axon activation. `activation` functions
  # are all element-wise with 1 input. Optionally has activation options.
  def decode_node(axon, node, _opts) do
    %Node{attribute: attrs, input: [inp], output: [output_name, op_type: op_type]} = node
    inp = layer!(axon, inp)
    attrs = options!(attrs)
    
    {op, opts} = if Map.has_key?(@op_decoder, op_type) do
      Map.get(@op_decoder, op_type)
    else
      raise ArgumentError, "activation #{op_type} is not supported"
    end

    opts =
      Enum.map(opts, fn {k, {name, default}} ->
        if attrs[name] do
          {k, attrs[name]}
        else
          {k, default}
        end
      end)

    opts = [name: output_name] ++ opts
    Axon.activation(inp, activation, opts)
  end
end
