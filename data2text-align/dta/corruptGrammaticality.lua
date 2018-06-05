--
-- Created by IntelliJ IDEA.
-- User: lperez
-- Date: 22/05/17
-- Time: 18:34
-- To change this template use File | Settings | File Templates.
--


function gramticalJudgementModel(opt, sentenceEncoder)
    --[[sentenceEncoder: computes the encoding of a given sequence of words and returns the sequence of output states.]]
    --local gramJudModel = nn.Sequential()
    --gramJudModel:add(sentenceEncoder)
    local gramJudModel = sentenceEncoder:clone('weight', 'bias', 'gradWeight', 'gradBias')
    gramJudModel:add(localize(nn.Select(2,-1))) --take the last output
    gramJudModel:add(localize(nn.Linear(opt.rnn_size,1)))
    gramJudModel:add(localize(nn.Sigmoid()))
    return gramJudModel
end

function multitaskCriterion()
    return localize(nn.BCECriterion())
end

function multitaskTrainBatch(classifier, criterion, batch, optim_func, optim_state)
  local xSet, ySet = unpack(batch)
  local x, dl_dx = classifier:getParameters()

  local feval = function(x_new)
      -- set x to x_new, if different
      -- (in this simple example, x_new will typically always point to x,
      -- so this copy is never happening)
      if x ~= x_new then
          x:copy(x_new)
      end

      dl_dx:zero()

      -- evaluate the loss function and its derivative wrt x, for that sample
      local criterion_scores = classifier:forward(xSet)

      local loss_x = criterion:forward(criterion_scores, ySet)

      --io.stdout:write('Loss: '..loss_x.."\n" )
      classifier:backward(xSet, criterion:backward(criterion_scores, ySet))

      return loss_x, dl_dx
  end

  local optim_x, ls = optim_func(feval, x, optim_state)

  return ls[1] --return batch loss
end

