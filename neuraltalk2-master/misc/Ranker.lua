require 'nn'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'

-------------------------------------------------------------------------------
-- Ranker Model
-------------------------------------------------------------------------------

local layer, parent = torch.class('nn.Ranker', 'nn.Module')
function layer:__init(opt)
  parent.__init(self)

  self.vocab_size = utils.getopt(opt, 'vocab_size') -- required
  self.input_encoding_size = utils.getopt(opt, 'input_encoding_size')
  self.seq_length = utils.getopt(opt, 'seq_length')
  self.linear_module = nn.Linear(self.vocab_size + 1, self.input_encoding_size)
end

function layer:getModulesList()
  return {self.linear_module}
end

function layer:parameters()
  local p,g = self.linear_module:parameters()

  local params = {}
  for k,v in pairs(p) do table.insert(params, v) end

  local grad_params = {}
  for k,v in pairs(g) do table.insert(grad_params, v) end

  return params, grad_params
end

--[[
input is a tuple of:
1. imgs: torch.Tensor of size NxK (K is dim of image code)
2. logprobs: torch.Tensor of size (D+2)xNx(M+1) (normalized log probabilities
for the next token at every iteration of the LSTM)
3. seq: DxN

returns a tuple of:
1. sim_matrix: a NxN Tensor
2. sembed: NxK
--]]
function layer:updateOutput(input)
  local imgs = input[1]
  local logprobs = input[2]
  local seq = input[3]
  local batch_size = imgs:size(1)
  local embed_size = imgs:size(2)

  local L, N, M1 = logprobs:size(1),logprobs:size(2), logprobs:size(3)
  local D = seq:size(1)
  assert(D == L-2, 'input Tensor should be 2 larger in time')
  local mask = utils.createTensor(logprobs:type(), logprobs:size())
  mask[{{2,L-1}}] = torch.expand(torch.reshape(torch.gt(seq, 0):type(mask:type()), D, N, 1), D, N, M1)

  local probs = torch.exp(logprobs) -- (D+2)xNx(M+1)
  local probs_mask = torch.cmul(probs, mask)
  local sum = torch.squeeze(torch.sum(probs_mask, 1))-- Nx(M+1)
  local sembed = self.linear_module:forward(sum)
  local sim_matrix = imgs * sembed:t()

  self.output = {sim_matrix, sembed}
  return self.output
end

--[[
input is a tuple of:
1. logprobs: torch.Tensor of size (D+2)xNx(M+1) (normalized log probabilities
for the next token at every iteration of the LSTM)
2. sembed: torch.Tensor of size NxK
3. imgs: torch.Tensor of size NxK (K is dim of image code)

returns a tuple of:
1. dlogprobs: (D+2)xNx(M+1)
2. dsembed: NxK
3. dimgs: NxK
--]]
function layer:updateGradInput(input, gradOutput)
  local dsim_matrix = gradOutput -- NxN
  local logprobs = input[1] -- (D+2)xNx(M+1)
  local sembed = input[2] -- NxK
  local imgs = input[3] -- NxK
  local seq = input[4] -- DxN
  local L, N, M1 = logprobs:size(1),logprobs:size(2), logprobs:size(3)
  local D = seq:size(1)
  assert(D == L-2, 'input Tensor should be 2 larger in time')

  local mask = utils.createTensor(logprobs:type(), logprobs:size())
  mask[{{2,L-1}}] = torch.expand(torch.reshape(torch.gt(seq, 0):type(mask:type()), D, N, 1), D, N, M1)
  local probs = torch.exp(logprobs) -- (D+2)xNx(M+1)
  local probs_mask = torch.cmul(probs, mask)
  local sum = torch.squeeze(torch.sum(probs_mask, 1))-- Nx(M+1)

  local dimgs = dsim_matrix * sembed
  local dsembed = dsim_matrix * imgs -- NxK
  local dsum = self.linear_module:backward(sum, dsembed) -- Nx(M+1)
  -- local dprobs = utils.createTensor(gradOutput:type(), probs:size()) -- (D+2)xNx(M+1)
  -- correct way to access weights?? 1,self.tmax or 2,self.tmax??
  -- dprobs[{{2,self.tmax}}] = torch.repeatTensor(dsum, self.tmax-1, 1, 1):div(self.tmax)
  local dprobs_mask = torch.repeatTensor(dsum, L, 1, 1)
  local dprobs = torch.cmul(dprobs_mask, mask)
  local dlogprobs = torch.cmul(dprobs, probs) -- (D+2)xNx(M+1)

  self.gradInput = {dlogprobs, dsembed, dimgs, torch.Tensor()}
  return self.gradInput
end


-------------------------------------------------------------------------------
-- Ranker Criterion
-------------------------------------------------------------------------------

local crit, parent = torch.class('nn.RankerCriterion', 'nn.Criterion')
function crit:__init()
  parent.__init(self)
end

--[[
input: similarity matrix, size NxN
--]]
function crit:updateOutput(input, seq)
  local sim_matrix = input
  N = sim_matrix:size(1)

  local ranking_loss = 0
  diag = utils.diag(sim_matrix)
  diag_row = torch.expand(torch.reshape(diag, N,1), N,N)
  diag_col = torch.expand(torch.reshape(diag, 1,N), N,N)

  -- Row loss
  margin_row = torch.cmax(sim_matrix-diag_row+1, 0)
  margin_row = margin_row - utils.diag(utils.diag(margin_row)) -- set diagonal to 0
  ranking_loss = ranking_loss + torch.sum(margin_row)

  -- Col loss
  margin_col = torch.cmax(sim_matrix-diag_col+1, 0)
  margin_col = margin_col - utils.diag(utils.diag(margin_col)) -- set diagonal to 0
  ranking_loss = ranking_loss + torch.sum(margin_col)

  ranking_loss = ranking_loss / N

  -- Compute similarity matrix gradients
  self.gradInput:resizeAs(input):zero():type(input:type()) -- reset to zeros
  margin_row_mask = torch.gt(margin_row, 0):type(input:type())
  self.gradInput = self.gradInput + margin_row_mask
  self.gradInput = self.gradInput - utils.diag(torch.sum(margin_row_mask, 2):view(-1))

  margin_col_mask = torch.gt(margin_col, 0):type(input:type())
  self.gradInput = self.gradInput + margin_col_mask
  self.gradInput = self.gradInput - utils.diag(torch.sum(margin_col_mask, 1):view(-1))

  self.gradInput:div(N)

  self.output = ranking_loss
  return self.output
end

function crit:updateGradInput(input, seq)
  return self.gradInput
end
