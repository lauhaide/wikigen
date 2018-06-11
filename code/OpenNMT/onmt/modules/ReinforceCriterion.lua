--
-- based on Facebook's ReinforceCriterion.lua
-- at https://github.com/facebookresearch/MIXER
--

local ReinforceCriterion, parent = torch.class('nn.ReinforceCriterion',
                                               'nn.Module')
-- This criterion implements the REINFORCE algorithm under the assumption that
-- the reward does not depend on the model parameters.
-- The constructor takes as input a function which is used to compute the reward
-- given the ground truth input sequence, the generated sequence and the current
-- time step.
-- The input to the criterion is a table whose entries are the output of the
-- RNN at a certain time step, namely:
-- (chosen_word, predicted_cumulative_reward)_t
-- It computes the total reward and bprop the derivative
-- w.r.t. the above provided inputs.
--  reward_func: user provided function to compute the reward
--   given ground truth, current sequence and current time step.
-- seq_length is the length of the sequence we use
-- skips is the number of time steps we skip from the input and target (init)
-- weight is the weight on the loss produced by this criterion
-- weight_predictive_reward is the weight on the gradient of the cumulative
--   reward predictor (only)
function ReinforceCriterion:__init(sampleStart, seqLen, batchSize)
   parent.__init(self)
   self.sampleStart = sampleStart
   self.cum_reward = torch.Tensor(batchSize, seqLen)
   self.grad_rf_sample = torch.Tensor(batchSize, seqLen)
   self.seqLen = seqLen
   self.grad_exp_reward = {}
   for t = 1, seqLen do
     self.grad_exp_reward[t] = torch.Tensor(batchSize, 1) --TODO is this loop limit and this size ok?
   end
end

function ReinforceCriterion:setSampleStart(start)
  self.sampleStart = start
end

function ReinforceCriterion:type(tp)
   parent.type(self, tp)
   self.cum_reward:type(tp)
   self.grad_rf_sample:type(tp)
   for t = 1, self.seqLen do
     self.grad_exp_reward[t]:type(tp)
   end
   
   return self
end

function ReinforceCriterion:set_weight(ww)
   self.weight = ww
end

function ReinforceCriterion:setSampleStart(start)
  self.sampleStart = start
end

function ReinforceCriterion:updateOutput(input)
  local rf_sample, exp_reward, true_reward, reward_mask = unpack(input)
  local Ty_rf = rf_sample:size(2)
  self.cum_reward:resizeAs(true_reward):zero()
  for t = Ty_rf - 1, self.sampleStart, -1 do
    self.cum_reward[{ {}, t }]:add( true_reward[{ {}, t+1 }], self.cum_reward[{ {}, t+1 }] )
    self.cum_reward[{ {}, t }]:cmul( reward_mask[{ {}, t+1 }] )
  end

  local seq_reward = self.cum_reward[{ {}, self.sampleStart }]
  local n_seq = seq_reward:ne(0):sum()

  return -seq_reward:sum() / rf_sample:size(1), n_seq
end

function ReinforceCriterion:updateGradInput(input)
  local rf_sample, exp_reward, true_reward, reward_mask = unpack(input)
  local Ty_rf = rf_sample:size(2)
  self.grad_rf_sample:resizeAs(rf_sample):zero()
  local grad_exp_reward = {}
  for t = Ty_rf - 1, self.sampleStart, -1 do
    self.grad_rf_sample[{ {}, t+1 }] = (exp_reward[t]:squeeze() - self.cum_reward[{ {}, t }])
    self.grad_rf_sample[{ {}, t+1 }]:div(rf_sample:size(1)) --batch normalisation
    self.grad_rf_sample[{ {}, t+1 }]:cmul( reward_mask[{ {}, t+1 }] )
    self.grad_exp_reward[t]:resizeAs(exp_reward[t]):zero()
    self.grad_exp_reward[t]:copy( self.grad_rf_sample[{ {}, t+1 }] )
    grad_exp_reward[t] = self.grad_exp_reward[t]
  end

  for t = 1, self.sampleStart - 1 do
    self.grad_exp_reward[t]:zero()
    grad_exp_reward[t] = self.grad_exp_reward[t]
  end

  self.gradInput = {self.grad_rf_sample, grad_exp_reward, exp_reward, self.cum_reward}

  return self.gradInput
end

function ReinforceCriterion:get_num_samples(input, target)
   return self.reward_func:num_samples(target, input)
end

function ReinforceCriterion:reset_reward()
    return self.reward_func:reset_vars()
end

function ReinforceCriterion:get_corpus_score()
    return self.reward_func:get_corpus_score()
end

function ReinforceCriterion:get_counts_corpus(target, pred)
    return self.reward_func:get_counts_corpus(target, pred)
end

function ReinforceCriterion:training_mode()
    self.reward_func:training_mode()
end

function ReinforceCriterion:test_mode()
    self.reward_func:test_mode()
end
