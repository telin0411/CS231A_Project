--[[
Unit tests for the LanguageModel implementation, making sure
that nothing crashes, that we can overfit a small dataset
and that everything gradient checks.
--]]

require 'torch'
require 'misc.LanguageModel'
require 'misc.Ranker'

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
local function forwardApiTestFactory(dtype)
  if dtype == 'torch.CudaTensor' then
    require 'cutorch'
    require 'cunn'
  end
  local function f()
    -- create LanguageModel instance
    local lmOpt = {}
    lmOpt.vocab_size = 5
    lmOpt.input_encoding_size = 11
    lmOpt.rnn_size = 8
    lmOpt.num_layers = 2
    lmOpt.dropout = 0
    lmOpt.seq_length = 7
    lmOpt.batch_size = 10
    local rankerOpt = {}
    rankerOpt.vocab_size = lmOpt.vocab_size
    rankerOpt.input_encoding_size = lmOpt.input_encoding_size
    rankerOpt.seq_length = lmOpt.seq_length
    local lm = nn.LanguageModel(lmOpt)
    local ranker = nn.Ranker(rankerOpt)
    local lmCrit = nn.LanguageModelCriterion()
    local rankerCrit = nn.RankerCriterion()
    lm:type(dtype)
    ranker:type(dtype)
    lmCrit:type(dtype)
    rankerCrit:type(dtype)

    -- construct some input to feed in
    local seq = torch.LongTensor(lmOpt.seq_length, lmOpt.batch_size):random(lmOpt.vocab_size)
    -- make sure seq can be padded with zeroes and that things work ok
    seq[{ {4, 7}, 1 }] = 0
    seq[{ {5, 7}, 6 }] = 0
    local imgs = torch.randn(lmOpt.batch_size, lmOpt.input_encoding_size):type(dtype)

    -- forward
    local logprobs = lm:forward{imgs, seq}
    tester:assertlt(torch.max(logprobs:view(-1)), 0) -- log probs should be <0
    local sim_matrix, sembed = unpack(ranker:forward{imgs, logprobs, lm.tmax})

    -- the output should be of size (seq_length + 2, batch_size, vocab_size + 1)
    -- where the +1 is for the special END token appended at the end.
    tester:assertTensorSizeEq(logprobs, {lmOpt.seq_length+2, lmOpt.batch_size, lmOpt.vocab_size+1})
    tester:assertTensorSizeEq(sim_matrix, {lmOpt.batch_size, lmOpt.batch_size})
    tester:assertTensorSizeEq(sembed, {lmOpt.batch_size, lmOpt.input_encoding_size})

    local loss_softmax = lmCrit:forward(logprobs, seq)
    local loss_ranking = rankerCrit:forward(sim_matrix, torch.Tensor())
    local loss = loss_softmax + loss_ranking

    -- backward
    local dsim_matrix = rankerCrit:backward(sim_matrix, torch.Tensor())
    local dlogprobs_lm = lmCrit:backward(logprobs, seq)
    local dlogprobs_ranker, dsembed, dexpanded_feats_ranker =
        unpack(ranker:backward({logprobs, sembed, imgs}, dsim_matrix))

    tester:assertTensorSizeEq(dlogprobs_lm, {lmOpt.seq_length+2, lmOpt.batch_size, lmOpt.vocab_size+1})
    tester:assertTensorSizeEq(dlogprobs_ranker, {lmOpt.seq_length+2, lmOpt.batch_size, lmOpt.vocab_size+1})
    tester:assertTensorSizeEq(dsembed, {lmOpt.batch_size, lmOpt.input_encoding_size})
    tester:assertTensorSizeEq(dexpanded_feats_ranker, {lmOpt.batch_size, lmOpt.input_encoding_size})

    local gradOutput = dlogprobs_lm + dlogprobs_ranker

    -- make sure the pattern of zero gradients is as expected
    local gradAbs = torch.max(torch.abs(dlogprobs_lm), 3):view(lmOpt.seq_length+2, lmOpt.batch_size)
    local gradZeroMask = torch.eq(gradAbs,0)
    local expectedGradZeroMask = torch.ByteTensor(lmOpt.seq_length+2,lmOpt.batch_size):zero()
    expectedGradZeroMask[{ {1}, {} }]:fill(1) -- first time step should be zero grad (img was passed in)
    expectedGradZeroMask[{ {6,9}, 1 }]:fill(1)
    expectedGradZeroMask[{ {7,9}, 6 }]:fill(1)
    tester:assertTensorEq(gradZeroMask:float(), expectedGradZeroMask:float(), 1e-8)

    local gradInput = lm:backward({imgs, seq}, gradOutput)
    tester:assertTensorSizeEq(gradInput[1], {lmOpt.batch_size, lmOpt.input_encoding_size})
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
  opt.batch_size = 20

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

--[[
local function gradCheckRanker()
  local dtype = 'torch.DoubleTensor'
  local lmOpt = {}
  lmOpt.vocab_size = 5
  lmOpt.input_encoding_size = 4
  lmOpt.rnn_size = 8
  lmOpt.num_layers = 2
  lmOpt.dropout = 0
  lmOpt.seq_length = 7
  lmOpt.batch_size = 6
  local rankerOpt = {}
  rankerOpt.vocab_size = lmOpt.vocab_size
  rankerOpt.input_encoding_size = lmOpt.input_encoding_size
  rankerOpt.seq_length = lmOpt.seq_length
  local lm = nn.LanguageModel(lmOpt)
  local ranker = nn.Ranker(rankerOpt)
  lm:type(dtype)
  ranker:type(dtype)

  local seq = torch.LongTensor(lmOpt.seq_length, lmOpt.batch_size):random(lmOpt.vocab_size)
  seq[{ {4, 7}, 1 }] = 0
  seq[{ {5, 7}, 4 }] = 0
  local imgs = torch.randn(lmOpt.batch_size, lmOpt.input_encoding_size):type(dtype)

  local logprobs = lm:forward{imgs, seq}
  local sim_matrix, sembed = unpack(ranker:forward{imgs, logprobs, lm.tmax})

  dsim_matrix = torch.DoubleTensor(sim_matrix:size(1), sim_matrix:size(2)):fill(1)
  local gradInput, dsembed, dexpanded_feats_ranker =
    unpack(ranker:backward({logprobs, sembed, imgs}, dsim_matrix))

  local function f(x)
    local sim_matrix, sembed = unpack(ranker:forward{imgs, x, lm.tmax})
    return sim_matrix
  end

  local gradInput_num = gradcheck.numeric_gradient(f, logprobs, 1.0, 1e-6)

  print(gradInput:size())
  print(gradInput_num:size())

  tester:assertTensorEq(gradInput, gradInput_num, 1e-4)
  tester:assertlt(gradcheck.relative_error(gradInput, gradInput_num, 1e-8), 1e-4)
end
]]--

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

  local seq = torch.LongTensor(opt.seq_length, opt.batch_size):random(opt.vocab_size)
  seq[{ {4, 7}, 1 }] = 0
  seq[{ {5, 7}, 4 }] = 0
  local imgs = torch.randn(opt.batch_size, opt.input_encoding_size):type(dtype)

  -- evaluate the analytic gradient
  local output = lm:forward{imgs, seq}
  local loss = crit:forward(output, seq)
  local gradOutput = crit:backward(output, seq)
  local gradInput, dummy = unpack(lm:backward({imgs, seq}, gradOutput))

  -- create a loss function wrapper
  local function f(x)
    local output = lm:forward{x, seq}
    local loss = crit:forward(output, seq)
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
  tester:assertlt(gradcheck.relative_error(gradInput, gradInput_num, 1e-8), 5e-4)
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

  local seq = torch.LongTensor(opt.seq_length, opt.batch_size):random(opt.vocab_size)
  seq[{ {4, 7}, 1 }] = 0
  seq[{ {5, 7}, 4 }] = 0
  local imgs = torch.randn(opt.batch_size, opt.input_encoding_size):type(dtype)

  local params, grad_params = lm:getParameters()
  print('number of parameters:', params:nElement(), grad_params:nElement())
  local lstm_params = 4*(opt.input_encoding_size + opt.rnn_size)*opt.rnn_size + opt.rnn_size*4*2
  local output_params = opt.rnn_size * (opt.vocab_size + 1) + opt.vocab_size+1
  local table_params = (opt.vocab_size + 1) * opt.input_encoding_size
  local expected_params = lstm_params + output_params + table_params
  print('expected:', expected_params)

  local function lossFun()
    grad_params:zero()
    local output = lm:forward{imgs, seq}
    local loss = crit:forward(output, seq)
    local gradOutput = crit:backward(output, seq)
    lm:backward({imgs, seq}, gradOutput)
    return loss
  end

  local loss
  local grad_cache = grad_params:clone():fill(1e-8)
  print('trying to overfit the language model on toy data:')
  for t=1,30 do
    loss = lossFun()
    -- test that initial loss makes sense
    if t == 1 then tester:assertlt(math.abs(math.log(opt.vocab_size+1) - loss), 0.1) end
    grad_cache:addcmul(1, grad_params, grad_params)
    params:addcdiv(-1e-1, grad_params, torch.sqrt(grad_cache)) -- adagrad update
    print(string.format('iteration %d/30: loss %f', t, loss))
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
tests.gradCheckRanker = gradCheckRanker
-- tests.gradCheck = gradCheck
-- tests.gradCheckLM = gradCheckLM
-- tests.overfit = overfit
-- tests.sample = sample
-- tests.sample_beam = sample_beam

tester:add(tests)
tester:run()
