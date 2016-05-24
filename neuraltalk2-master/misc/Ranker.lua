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
  self.lookup_table = nn.LookupTable(self.vocab_size + 1, self.input_encoding_size)
end

--[[
input is a tuple of:
1. imgs: torch.Tensor of size NxK (K is dim of image code)
2. logprobs: torch.Tensor of size (D+2)xNx(M+1) (normalized log probabilities
for the next token at every iteration of the LSTM)

returns a NxN Tensor (similarity matrix)
--]]
function layer:updateOutput(input)
  local imgs = input[1]
  local logprobs = input[2]
  local batch_size = imgs:size(1)
  local embed_size = imgs:size(2)

  sentence_embed = torch.DoubleTensor(batch_size, embed_size):zero()
  for t=1,self.seq_length+2 do
    logprob_t = logprobs[t] -- Nx(M+1)
    sampleLogprobs, it = torch.max(logprob_t, 2)
    it = it:view(-1):long()
    xt = self.lookup_table:forward(it) -- what's the size of xt??
    sentence_embed = sentence_embed + xt
  end
end

function layer:updateGradInput(input, gradOutput)
end


-------------------------------------------------------------------------------
-- Ranker Criterion
-------------------------------------------------------------------------------

local crit, parent = torch.class('nn.RankerCriterion', 'nn.Criterion')
function crit:__init()
  parent.__init(self)
end
