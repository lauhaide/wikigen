--require 'nn'

local StupidMaxThing, parent = torch.class('nn.StupidMaxThing', 'nn.Module')

function StupidMaxThing:__init()
    parent.__init(self)
    self.gradInput = {}
end

function StupidMaxThing:updateOutput(input)
    local probs, srcwords = input[1], input[2]
    self.output:resizeAs(probs):zero()
    local tidxs, tvals = {}, {}

    -- get max for each word type
    for i = 1, probs:size(1) do
        local max_vals, max_idxs = {}, {}
        for j = 1, probs:size(2) do
            local wrd = srcwords[i][j]
            if not max_vals[wrd] or probs[i][j] > max_vals[wrd] then
                max_vals[wrd] = probs[i][j]
                max_idxs[wrd] = j
            end
        end

        for k,v in pairs(max_vals) do
            table.insert(tvals, v)
            table.insert(tidxs, (i-1)*probs:size(2)+max_idxs[k])
        end
    end

    self.maxidxs = torch.CudaTensor(tidxs)
    local maxvals = torch.CudaTensor(tvals)
    self.output:view(-1):indexCopy(1, self.maxidxs, maxvals)
    return self.output
end

function StupidMaxThing:updateGradInput(input, gradOutput)
    local probs, srcwords = input[1], input[2]
    self.gradInput[1] = self.gradInput[1] or probs.new()

    self.gradInput[1]:resizeAs(probs):zero()
    self.gradInput[1]:view(-1):indexFill(1, self.maxidxs, 1)
    self.gradInput[1]:cmul(gradOutput)
    return self.gradInput
end


-- mlp = nn.Sequential()
--         :add(nn.ParallelTable()
--               :add(nn.Linear(5,6))
--               :add(nn.Identity()))
--         :add(nn.StupidMaxThing())
--         :add(nn.CMul(6))
--         :add(nn.Sum(2))
--
--
--
-- myx = torch.randn(2, 5)
-- myy = torch.randn(2,1)
-- mysrcidxs = torch.LongTensor({{2,3,3,3,3,1},{1,4,4,4,4,4}})
-- crit = nn.MSECriterion()
--
-- feval = function(x)
--     return crit:forward(mlp:forward({x,mysrcidxs}), myy)
-- end
--
-- crit:forward(mlp:forward({myx,mysrcidxs}), myy)
-- dpdc = crit:backward(mlp.output, myy)
-- mlp:backward({myx, mysrcidxs}, dpdc)
--
-- eps = 1e-5
--
-- for i = 1, myx:size(1) do
--     for j = 1, myx:size(2) do
--         local orig  = myx[i][j]
--         myx[i][j] = myx[i][j] + eps
--         local rloss = feval(myx)
--         myx[i][j] = myx[i][j] - 2*eps
--         local lloss = feval(myx)
--         local fd = (rloss - lloss)/(2*eps)
--         print(fd, mlp.gradInput[1][i][j])
--         myx[i][j] = orig
--     end
-- end
