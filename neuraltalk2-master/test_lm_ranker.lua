--[[
Unit tests for the LanguageModel implementation, making sure
that nothing crashes, that we can overfit a small dataset
and that everything gradient checks.
--]]

require 'torch'
require 'misc.LanguageModel'
require 'misc.Ranker'
local utils = require 'misc.utils'

local gradcheck = require 'misc.gradcheck'

local tests = {}
local tester = torch.Tester()

-- validates the size and dimensions of a given
-- tensor a to be size given in table sz
function tester:assertTensorSizeEq(a, sz)
  tester:asserteq(a:nDimension(), #sz)
  for i=1,#sz do
    tester:asserteq(a:size(i), sz[i])
  end
end

-- Test the API of the Language Model
-- And Ranker Model
local function forwardApiTestFactory(dtype)
  if dtype == 'torch.CudaTensor' then
    require 'cutorch'
    require 'cunn'
  end
  local function f()
    -- create LanguageModel instance
    local opt = {}
    opt.vocab_size = 5
    opt.input_encoding_size = 11
    opt.rnn_size = 8
    opt.num_layers = 2
    opt.dropout = 0
    opt.seq_length = 7
    opt.batch_size = 10
    local lm = nn.LanguageModel(opt)
    local crit = nn.LanguageModelCriterion()
    lm:type(dtype)
    crit:type(dtype)

    -- create Ranker instance
    local ranker = nn.Ranker(opt)
    local crit_ranker = nn.RankerCriterion()
    ranker:type(dtype)
    crit_ranker:type(dtype)

    -- construct some input to feed in
    local seq = torch.LongTensor(opt.seq_length, opt.batch_size):random(opt.vocab_size)
    -- make sure seq can be padded with zeroes and that things work ok
    seq[{ {4, 7}, 1 }] = 0
    seq[{ {5, 7}, 6 }] = 0
    local imgs = torch.randn(opt.batch_size, opt.input_encoding_size):type(dtype)
    local output = lm:forward{imgs, seq}
    tester:assertlt(torch.max(output:view(-1)), 0) -- log probs should be <0

    local sim_matrix, sembed = unpack(ranker:forward{imgs, output, lm.tmax})

    -- the output should be of size (seq_length + 2, batch_size, vocab_size + 1)
    -- where the +1 is for the special END token appended at the end.
    tester:assertTensorSizeEq(output, {opt.seq_length+2, opt.batch_size, opt.vocab_size+1})
    tester:assertTensorSizeEq(sim_matrix, {opt.batch_size, opt.batch_size})
    tester:assertTensorSizeEq(sembed, {opt.batch_size, opt.input_encoding_size})

    local loss = crit:forward(output, seq)
    local ranking_loss = crit_ranker:forward(sim_matrix, torch.Tensor())

    local gradOutput = crit:backward(output, seq)
    local dsim_matrix = crit_ranker:backward(sim_matrix, torch.Tensor())
    tester:assertTensorSizeEq(gradOutput, {opt.seq_length+2, opt.batch_size, opt.vocab_size+1})
    tester:assertTensorSizeEq(dsim_matrix, {opt.batch_size, opt.batch_size})

    -- make sure the pattern of zero gradients is as expected
    local gradAbs = torch.max(torch.abs(gradOutput), 3):view(opt.seq_length+2, opt.batch_size)
    local gradZeroMask = torch.eq(gradAbs,0)
    local expectedGradZeroMask = torch.ByteTensor(opt.seq_length+2,opt.batch_size):zero()
    expectedGradZeroMask[{ {1}, {} }]:fill(1) -- first time step should be zero grad (img was passed in)
    expectedGradZeroMask[{ {6,9}, 1 }]:fill(1)
    expectedGradZeroMask[{ {7,9}, 6 }]:fill(1)
    tester:assertTensorEq(gradZeroMask:float(), expectedGradZeroMask:float(), 1e-8)

    local dlogprobs_ranker, dsembed, dimgs_ranker =
      unpack(ranker:backward({output, sembed, imgs}, dsim_matrix))
    gradOutput = torch.add(gradOutput, dlogprobs_ranker)
    local gradInput = lm:backward({imgs, seq}, gradOutput)
    gradInput[1] = torch.add(gradInput[1], dimgs_ranker)
    tester:assertTensorSizeEq(gradInput[1], {opt.batch_size, opt.input_encoding_size})
    tester:asserteq(gradInput[2]:nElement(), 0, 'grad on seq should be empty tensor')

  end
  return f
end

-- test just the language model alone (without the criterion)
local function gradCheckLM()

  local dtype = 'torch.DoubleTensor'
  local opt = {}
  opt.vocab_size = 5
  opt.input_encoding_size = 4
  opt.rnn_size = 8
  opt.num_layers = 2
  opt.dropout = 0
  opt.seq_length = 7
  opt.batch_size = 6
  local lm = nn.LanguageModel(opt)
  local crit = nn.LanguageModelCriterion()
  lm:type(dtype)
  crit:type(dtype)

  local seq = torch.LongTensor(opt.seq_length, opt.batch_size):random(opt.vocab_size)
  seq[{ {4, 7}, 1 }] = 0
  seq[{ {5, 7}, 4 }] = 0
  local imgs = torch.randn(opt.batch_size, opt.input_encoding_size):type(dtype)

  -- evaluate the analytic gradient
  local output = lm:forward{imgs, seq}
  local w = torch.randn(output:size(1), output:size(2), output:size(3))

  -- generate random weighted sum criterion
  local loss = torch.sum(torch.cmul(output, w))
  local gradOutput = w
  local gradInput, dummy = unpack(lm:backward({imgs, seq}, gradOutput))

  -- create a loss function wrapper
  local function f(x)
    local output = lm:forward{x, seq}
    local loss = torch.sum(torch.cmul(output, w))
    return loss
  end

  local gradInput_num = gradcheck.numeric_gradient(f, imgs, 1, 1e-6)

  -- print(gradInput)
  -- print(gradInput_num)
  -- local g = gradInput:view(-1)
  -- local gn = gradInput_num:view(-1)
  -- for i=1,g:nElement() do
  --   local r = gradcheck.relative_error(g[i],gn[i])
  --   print(i, g[i], gn[i], r)
  -- end

  tester:assertTensorEq(gradInput, gradInput_num, 1e-4)
  tester:assertlt(gradcheck.relative_error(gradInput, gradInput_num, 1e-8), 1e-4)
end

local function gradCheckRankerLoss()
  local dtype = 'torch.DoubleTensor'
  local opt = {}
  opt.vocab_size = 5
  opt.input_encoding_size = 4
  opt.rnn_size = 8
  opt.num_layers = 2
  opt.dropout = 0
  opt.seq_length = 7
  opt.batch_size = 6

  -- create Ranker instance
  local ranker = nn.Ranker(opt)
  local crit_ranker = nn.RankerCriterion()
  ranker:type(dtype)
  crit_ranker:type(dtype)

  local sim_matrix = torch.randn(opt.batch_size, opt.batch_size):type(dtype)
  local ranking_loss = crit_ranker:forward(sim_matrix, torch.Tensor())
  local dsim_matrix = crit_ranker:backward(sim_matrix, torch.Tensor())

  local function f(x)
    local loss = crit_ranker:forward(x, torch.Tensor())
    return loss
  end

  dsim_matrix_cp = dsim_matrix:clone()
  local dsim_matrix_num = gradcheck.numeric_gradient(f, sim_matrix, 1, 1e-6)
  dsim_matrix = dsim_matrix_cp
  tester:assertTensorEq(dsim_matrix, dsim_matrix_num, 1e-4)
  tester:assertlt(gradcheck.relative_error(dsim_matrix, dsim_matrix_num, 1e-8), 1e-4)
end

local function gradCheck()
  local dtype = 'torch.DoubleTensor'
  local opt = {}
  opt.vocab_size = 5
  opt.input_encoding_size = 4
  opt.rnn_size = 8
  opt.num_layers = 2
  opt.dropout = 0
  opt.seq_length = 7
  opt.batch_size = 6
  local lm = nn.LanguageModel(opt)
  local crit = nn.LanguageModelCriterion()
  lm:type(dtype)
  crit:type(dtype)

  -- create Ranker instance
  local ranker = nn.Ranker(opt)
  local crit_ranker = nn.RankerCriterion()
  ranker:type(dtype)
  crit_ranker:type(dtype)

  local seq = torch.LongTensor(opt.seq_length, opt.batch_size):random(opt.vocab_size)
  seq[{ {4, 7}, 1 }] = 0
  seq[{ {5, 7}, 4 }] = 0
  local imgs = torch.randn(opt.batch_size, opt.input_encoding_size):type(dtype)

  -- evaluate the analytic gradient
  local output = lm:forward{imgs, seq}
  local loss = crit:forward(output, seq)
  local sim_matrix, sembed = unpack(ranker:forward{imgs, output, seq})
  local ranking_loss = crit_ranker:forward(sim_matrix, torch.Tensor())
  local gradOutput = crit:backward(output, seq)
  local dsim_matrix = crit_ranker:backward(sim_matrix, torch.Tensor())
  local dlogprobs_ranker, dsembed, dimgs_ranker, dummy =
    unpack(ranker:backward({output, sembed, imgs, seq}, dsim_matrix))
  gradOutput = torch.add(gradOutput, dlogprobs_ranker)
  local gradInput, dummy = unpack(lm:backward({imgs, seq}, gradOutput))
  gradInput = torch.add(gradInput, dimgs_ranker)

  -- create a loss function wrapper
  local function f(x)
    local output = lm:forward{x, seq}
    local loss = crit:forward(output, seq)
    local sim_matrix, sembed = unpack(ranker:forward{x, output, seq})
    local ranking_loss = crit_ranker:forward(sim_matrix, torch.Tensor())
    loss = loss + ranking_loss
    return loss
  end

  local gradInput_cp = gradInput:clone()
  local gradInput_num = gradcheck.numeric_gradient(f, imgs, 1, 1e-6)
  gradInput = gradInput_cp

  -- print(gradInput)
  -- print(gradInput_num)
  -- local g = gradInput:view(-1)
  -- local gn = gradInput_num:view(-1)
  -- for i=1,g:nElement() do
  --   local r = gradcheck.relative_error(g[i],gn[i])
  --   print(i, g[i], gn[i], r)
  -- end

  tester:assertTensorEq(gradInput, gradInput_num, 1e-4)
  -- tester:assertlt(gradcheck.relative_error(gradInput, gradInput_num, 1e-8), 5e-4)
end

local function overfit()
  local dtype = 'torch.DoubleTensor'
  local opt = {}
  opt.vocab_size = 5
  opt.input_encoding_size = 7
  opt.rnn_size = 24
  opt.num_layers = 1
  opt.dropout = 0
  opt.seq_length = 7
  opt.batch_size = 6
  local lm = nn.LanguageModel(opt)
  local crit = nn.LanguageModelCriterion()
  lm:type(dtype)
  crit:type(dtype)

  -- create Ranker instance
  local ranker = nn.Ranker(opt)
  local crit_ranker = nn.RankerCriterion()
  ranker:type(dtype)
  crit_ranker:type(dtype)

  local seq = torch.LongTensor(opt.seq_length, opt.batch_size):random(opt.vocab_size)
  seq[{ {4, 7}, 1 }] = 0
  seq[{ {5, 7}, 4 }] = 0
  local imgs = torch.randn(opt.batch_size, opt.input_encoding_size):type(dtype)

  local params, grad_params = lm:getParameters()
  local ranker_params, ranker_grad_params = ranker:getParameters()
  print('number of parameters:', params:nElement()+ranker_params:nElement(), grad_params:nElement()+ranker_grad_params:nElement())
  local lstm_params = 4*(opt.input_encoding_size + opt.rnn_size)*opt.rnn_size + opt.rnn_size*4*2
  local output_params = opt.rnn_size * (opt.vocab_size + 1) + opt.vocab_size+1
  local table_params = (opt.vocab_size + 1) * opt.input_encoding_size
  local ranker_linear_params = (opt.vocab_size + 1) * opt.input_encoding_size
  local expected_params = lstm_params + output_params + table_params + ranker_linear_params
  print('expected:', expected_params)

  local reg_ranker = 1
  local reg_softmax = 0.1
  local function lossFun()
    grad_params:zero()
    local output = lm:forward{imgs, seq}
    local softmax_loss = crit:forward(output, seq)
    softmax_loss = softmax_loss * reg_softmax
    local sim_matrix, sembed = unpack(ranker:forward{imgs, output, seq})
    local ranking_loss = crit_ranker:forward(sim_matrix, torch.Tensor())
    ranking_loss = ranking_loss * reg_ranker
    local gradOutput = crit:backward(output, seq)
    gradOutput = gradOutput * reg_softmax
    local dsim_matrix = crit_ranker:backward(sim_matrix, torch.Tensor())
    dsim_matrix = dsim_matrix * reg_ranker
    local dlogprobs_ranker, dsembed, dimgs_ranker, dummy =
      unpack(ranker:backward({output, sembed, imgs, seq}, dsim_matrix))
    gradOutput = torch.add(gradOutput, dlogprobs_ranker)
    local loss = softmax_loss + ranking_loss
    lm:backward({imgs, seq}, gradOutput)
    return {loss, softmax_loss, ranking_loss}
  end

  local loss
  local grad_cache = grad_params:clone():fill(1e-8)
  local ranker_grad_cache = ranker_grad_params:clone():fill(1e-8)
  print('trying to overfit the language model on toy data:')
  local t = 0
  loss = lossFun()
  while loss[1] > 1 do
  -- for t=1,1000 do
    t = t + 1
    loss = lossFun()
    -- test that initial loss makes sense
    if t == 1 then tester:assertlt(math.abs(math.log(opt.vocab_size+1) - loss[1]), 0.1) end
    grad_cache:addcmul(1, grad_params, grad_params)
    ranker_grad_cache:addcmul(1, ranker_grad_params, ranker_grad_params)
    params:addcdiv(-1e-1, grad_params, torch.sqrt(grad_cache)) -- adagrad update
    ranker_params:addcdiv(-1e-2, ranker_grad_params, torch.sqrt(ranker_grad_cache)) -- adagrad update
    print(string.format('iteration %d/30: loss %f softmax %f ranking %f', t, loss[1], loss[2], loss[3]))
  end
  -- holy crap adagrad destroys the loss function!

  tester:assertlt(loss, 0.2)
end

-- check that we can call :sample() and that correct-looking things happen
local function sample()
  local dtype = 'torch.DoubleTensor'
  local opt = {}
  opt.vocab_size = 5
  opt.input_encoding_size = 4
  opt.rnn_size = 8
  opt.num_layers = 2
  opt.dropout = 0
  opt.seq_length = 7
  opt.batch_size = 6
  local lm = nn.LanguageModel(opt)

  local imgs = torch.randn(opt.batch_size, opt.input_encoding_size):type(dtype)
  local seq = lm:sample(imgs)

  tester:assertTensorSizeEq(seq, {opt.seq_length, opt.batch_size})
  tester:asserteq(seq:type(), 'torch.LongTensor')
  tester:assertge(torch.min(seq), 1)
  tester:assertle(torch.max(seq), opt.vocab_size+1)
  print('\nsampled sequence:')
  print(seq)
end


-- check that we can call :sample_beam() and that correct-looking things happen
-- these are not very exhaustive tests and basic sanity checks
local function sample_beam()
  local dtype = 'torch.DoubleTensor'
  torch.manualSeed(1)

  local opt = {}
  opt.vocab_size = 10
  opt.input_encoding_size = 4
  opt.rnn_size = 8
  opt.num_layers = 1
  opt.dropout = 0
  opt.seq_length = 7
  opt.batch_size = 6
  local lm = nn.LanguageModel(opt)

  local imgs = torch.randn(opt.batch_size, opt.input_encoding_size):type(dtype)

  local seq_vanilla, logprobs_vanilla = lm:sample(imgs)
  local seq, logprobs = lm:sample(imgs, {beam_size = 1})

  -- check some basic I/O, types, etc.
  tester:assertTensorSizeEq(seq, {opt.seq_length, opt.batch_size})
  tester:asserteq(seq:type(), 'torch.LongTensor')
  tester:assertge(torch.min(seq), 0)
  tester:assertle(torch.max(seq), opt.vocab_size+1)

  -- doing beam search with beam size 1 should return exactly what we had before
  print('')
  print('vanilla sampling:')
  print(seq_vanilla)
  print('beam search sampling with beam size 1:')
  print(seq)
  tester:assertTensorEq(seq_vanilla, seq, 0) -- these are LongTensors, expect exact match
  tester:assertTensorEq(logprobs_vanilla, logprobs, 1e-6) -- logprobs too

  -- doing beam search with higher beam size should yield higher likelihood sequences
  local seq2, logprobs2 = lm:sample(imgs, {beam_size = 8})
  local logsum = torch.sum(logprobs, 1)
  local logsum2 = torch.sum(logprobs2, 1)
  print('')
  print('beam search sampling with beam size 1:')
  print(seq)
  print('beam search sampling with beam size 8:')
  print(seq2)
  print('logprobs:')
  print(logsum)
  print(logsum2)

  -- the logprobs should always be >=, since beam_search is better argmax inference
  tester:assert(torch.all(torch.gt(logsum2, logsum)))
end

-- tests.doubleApiForwardTest = forwardApiTestFactory('torch.DoubleTensor')
-- tests.floatApiForwardTest = forwardApiTestFactory('torch.FloatTensor')
-- tests.cudaApiForwardTest = forwardApiTestFactory('torch.CudaTensor')
-- tests.gradCheckRankerLoss = gradCheckRankerLoss
tests.gradCheck = gradCheck
-- tests.gradCheckLM = gradCheckLM
-- tests.overfit = overfit
tests.sample = sample
tests.sample_beam = sample_beam

tester:add(tests)
tester:run()
