require 'nn'
require 'rnn'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'

-------------------------------------------------------------------------------
-- Ranker Model
-------------------------------------------------------------------------------

local layer, parent = torch.class('nn.RankerLSTM', 'nn.Module')
function layer:__init(opt)
  parent.__init(self)

  self.vocab_size = utils.getopt(opt, 'vocab_size') -- required
  self.input_encoding_size = utils.getopt(opt, 'input_encoding_size')
  self.seq_length = utils.getopt(opt, 'seq_length')
  self.reg = utils.getopt(opt, 'reg_ranker')
  self.linear_module = nn.Linear(self.vocab_size + 1, self.input_encoding_size)
  self.seqLSTM = nn.SeqLSTM(self.input_encoding_size, self.input_encoding_size)
  self.seqLSTM.maskzero = true
end

function layer:getModulesList()
  return {self.linear_module, self.seqLSTM}
end

function layer:parameters()
  local p1,g1 = self.linear_module:parameters()
  local p2,g2 = self.seqLSTM:parameters()

  local params = {}
  for k,v in pairs(p1) do table.insert(params, v) end
  for k,v in pairs(p2) do table.insert(params, v) end

  local grad_params = {}
  for k,v in pairs(g1) do table.insert(grad_params, v) end
  for k,v in pairs(g2) do table.insert(grad_params, v) end

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
  local K = imgs:size(2)

  local L, N, M1 = logprobs:size(1),logprobs:size(2), logprobs:size(3)
  local D = seq:size(1)
  assert(D == L-2, 'input Tensor should be 2 larger in time')
  local mask = utils.createTensor(logprobs:type(), logprobs:size())
  mask[{{2,L-1}}] = torch.expand(torch.reshape(torch.gt(seq, 0):type(mask:type()), D, N, 1), D, N, M1)

  local probs = torch.exp(logprobs) -- (D+2)xNx(M+1)
  local probs_mask = torch.cmul(probs, mask)
  d = torch.LongStorage(3)
  d[1] = D; d[2] = N; d[3] = K
  local wembeds = utils.createTensor(logprobs:type(), d) -- DxNxK
  for t = 1,D do
    wembeds[t] = self.linear_module:forward(probs_mask[t+1])
  end

  -- take the last time step output as sentence embedding
  local sembed = self.seqLSTM:forward(wembeds)[D] -- NxK
  local sim_matrix = imgs * sembed:t()

  self.output = {sim_matrix, sembed, wembeds}
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
  local wembeds = input[3] -- DxNxK
  local imgs = input[4] -- NxK
  local seq = input[5] -- DxN
  local L, N, M1 = logprobs:size(1),logprobs:size(2), logprobs:size(3)
  local D = seq:size(1)
  assert(D == L-2, 'input Tensor should be 2 larger in time')

  local mask = utils.createTensor(logprobs:type(), logprobs:size())
  mask[{{2,L-1}}] = torch.expand(torch.reshape(torch.gt(seq, 0):type(mask:type()), D, N, 1), D, N, M1)
  local probs = torch.exp(logprobs) -- (D+2)xNx(M+1)
  local probs_mask = torch.cmul(probs, mask)
  local sum = torch.squeeze(torch.sum(probs_mask, 1))-- Nx(M+1)

  local dimgs = dsim_matrix * sembed
  local dsembed = dsim_matrix:t() * imgs -- NxK
  local dsembed_D = utils.createTensor(dsembed:type(), wembeds:size())
  dsembed_D[D] = dsembed
  local dwembeds = self.seqLSTM:backward(wembeds, dsembed_D) -- DxNxK
  local dprobs_mask = utils.createTensor(probs_mask:type(), probs_mask:size())
  for t = 1,D do
    dprobs_mask[t+1] = self.linear_module:backward(probs_mask[t+1], dwembeds[t])
  end
  local dprobs = torch.cmul(dprobs_mask, mask)
  local dlogprobs = torch.cmul(dprobs, probs) -- (D+2)xNx(M+1)

  self.gradInput = {dlogprobs, dsembed, dwembeds, dimgs, torch.Tensor()}
  return self.gradInput
end

-------------------------------------------------------------------------------
-- Ranker Criterion
-------------------------------------------------------------------------------

local crit, parent = torch.class('nn.RankerCriterion', 'nn.Criterion')
function crit:__init(opt)
  parent.__init(self)
  self.reg = utils.getopt(opt, 'reg_ranker')
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

  ranking_loss = self.reg * ranking_loss / N

  -- Compute similarity matrix gradients
  self.dsim_matrix = torch.Tensor(input:size()):type(input:type()):zero()
  margin_row_mask = torch.gt(margin_row, 0):type(input:type())
  self.dsim_matrix = self.dsim_matrix + margin_row_mask
  self.dsim_matrix = self.dsim_matrix - utils.diag(torch.sum(margin_row_mask, 2):view(-1))

  margin_col_mask = torch.gt(margin_col, 0):type(input:type())
  self.dsim_matrix = self.dsim_matrix + margin_col_mask
  self.dsim_matrix = self.dsim_matrix - utils.diag(torch.sum(margin_col_mask, 1):view(-1))

  self.dsim_matrix:div(N):mul(self.reg)

  self.output = ranking_loss
  return self.output
end

function crit:updateGradInput(input, seq)
  self.gradInput = self.dsim_matrix:clone()
  return self.gradInput
end
