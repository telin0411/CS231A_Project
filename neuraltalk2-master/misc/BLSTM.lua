require 'nn'
require 'rnn'

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
  self.outputs = self.core:forward(input)
  return self.outputs

function layer:updateGradInput(inputs, gradOutputs)
  self.gradInput = self.core:backward(inputs, gradOutputs)
  return self.gradInput
  
