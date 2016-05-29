require 'nn'
require 'rnn'
local utils = require 'misc.utils'

local layer, parent = torch.class('nn.BLSTM', 'nn.Module')

function layer:__init(opt)
  parent.__init(self)
  self.vocab_size = utils.getopt(opt, 'vocab_size') -- required
  self.input_encoding_size = utils.getopt(opt, 'input_encoding_size')
  local fwd = nn.LSTM(self.vocab_size + 1, self.input_encoding_size)
  local bwd = fwd:clone()
  bwd:reset()
  local merge = nn.CAddTable()
  local brnn = nn.BiSequencerLM(fwd, bwd, merge)
  print (brnn)
  self.core = brnn
end

function layer:getModulesList()
  return {self.core}
end

function layer:parameters()
  local p1,g1 = self.core:parameters()

  local params = {}
  for k,v in pairs(p1) do table.insert(params, v) end

  local grad_params = {}
  for k,v in pairs(g1) do table.insert(grad_params, v) end

  return params, grad_params
end

function layer:updateOutput(input)
  inputs = {}
  for i = 1, input:size(1) do
    inputs[i] = input[i]
  end 
  self.outputs = self.core:forward(inputs)
  return self.outputs
end

function layer:updateGradInput(input, gradOutput)
  gradOutputs = {}
  inputs      = {}
  for i = 1, gradOutput:size(1) do
    gradOutputs[i] = gradOutput[i]
  end
  for i = 1, input:size(1) do
    inputs[i] = input[i]
  end
  self.gradInput = self.core:backward(inputs, gradOutputs)
  return self.gradInput
end  
