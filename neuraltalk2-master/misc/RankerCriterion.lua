require 'nn'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'

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
