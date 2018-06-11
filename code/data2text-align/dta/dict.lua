--
-- Created by IntelliJ IDEA.
-- User: lperez
-- Date: 07/03/17
-- Time: 16:55
-- To change this template use File | Settings | File Templates.
--
-- Class adapted from https://github.com/jroakes/OpenNMT/blob/master/onmt/utils/dict.lua

local Dict = torch.class("Dict")

function Dict:__init(data)

  self.idxToLabel = {}
  self.labelToIdx = {}
  self:loadFile(data)

end

function Dict:loadFile(filename)
  local f = io.open(filename, 'r')

  for line in f:lines() do
    local fields = {}
    for w in line:gmatch '([^%s]+)' do
        table.insert(fields, w)
        end
    local label = fields[1]
    local idx = tonumber(fields[2])

    self:add(label, idx)
  end
  f:close()
end


--[[ Return the number of entries in the dictionary. ]]
function Dict:size()
  return #self.idxToLabel
end


--[[ Lookup `key` in the dictionary: it can be an index or a string. ]]
function Dict:lookup(key)
  if type(key) == "string" then
    return self.labelToIdx[key]
  else
    return self.idxToLabel[key]
  end
end


function Dict:add(label, idx)
  if idx ~= nil then
    self.idxToLabel[idx] = label
    self.labelToIdx[label] = idx
  end
end


--[[
  Convert `labels` to indices. Use `unkWord` if not found.
  Optionally insert `bosWord` at the beginning and `eosWord` at the end.
]]
function Dict:convertToIdx(labels, unkWord, bosWord, eosWord)
  local vec = {}

  if bosWord ~= nil then
    table.insert(vec, self:lookup(bosWord))
  end

  for i = 1, #labels do
    local idx = self:lookup(labels[i])
    if idx == nil then
      idx = self:lookup(unkWord)
    end
    table.insert(vec, idx)
  end

  if eosWord ~= nil then
    table.insert(vec, self:lookup(eosWord))
  end

  return torch.IntTensor(vec)
end


--[[ Convert `idx` to labels. If index `stop` is reached, convert it and return. ]]
function Dict:convertToLabels(idx, stop)
  local labels = {}

  for i = 1, #idx do
    table.insert(labels, self:lookup(idx[i]))
    if idx[i] == stop then
      break
    end
  end

  return labels
end

function Dict:getConstants()
    return {
  PAD = 0,
  UNK = 1,

  PAD_WORD = '<blank>',
  UNK_WORD = '<unk>'
}
end

return Dict