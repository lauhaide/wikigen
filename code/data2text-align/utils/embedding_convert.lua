require('torch')
local tds = require('tds')
local zlib = require ('zlib')
local path = require('pl.path')
require('../dta/dict')

local cmd = torch.CmdLine()


cmd:text("")
cmd:text("**embedding_convert.lua**")
cmd:text("")

cmd:text("")
cmd:text("**Data options**")
cmd:text("")

cmd:option('-dict_file', '', [[Path to outputted dict file from Preprocess.py.]])
cmd:option('-embed_file', '',[[Path to embedding file.]])
cmd:option('-out_folder', '',[[Output file path/label]])

cmd:text("")
cmd:text("**Embedding options**")
cmd:text("")


cmd:option('-embed_type', 'word2vec',[['word2vec' or 'glove'. Ignored if auto_lang is used.]])
cmd:option('-normalize', 'true',[[Boolean to normalize the word vectors, or not.]])
cmd:option('-report_every', '100000',[[Print stats every this many lines read from embedding file.]])

local opt = cmd:parse(arg)


local function loadEmbeddings(embeddingFilename, embeddingType, dictionary)


  --[[Converts binary to strings - Courtesy of https://github.com/rotmanmi/word2vec.torch]]
  local function readStringv2(file)
    local str = {}
    local max_w = 50

    for _ = 1, max_w do
      local char = file:readChar()

      if char == 32 or char == 10 or char == 0 then
        break
      else
        str[#str + 1] = char
      end
    end

    str = torch.CharStorage(str)
    return str:string()

  end

  -- [[Looks for cased version and then lower version of matching dictionary word.]]
  local function locateIdx(word, dict)

    local idx = nil

    if dict:lookup(word) ~= nil then
      idx = dict:lookup(word)

    elseif dict:lookup(word:lower()) ~= nil then
      idx = dict:lookup(word:lower())

    end

    return idx

  end


  -- [[Fills value for unmatched embeddings]]
  local function fillGaps(weights, loaded, dictSize, embeddingSize)

    for idx = 1, dictSize do
      if loaded[idx] == nil then
        for i=1, embeddingSize do
          weights[idx][i] = torch.uniform(-1, 1)
        end
      end
    end

    return weights

  end

  -- [[Initializes OpenNMT constants.]]
  local function preloadSpecial (weights, loaded, dict, embeddingSize)

    local specials = {dict:getConstants().UNK_WORD}

    for i = 1, #specials do
      local idx = locateIdx(specials[i], dict)
      for e=1, embeddingSize do
        weights[idx][e] = torch.normal(0, 0.9)
      end
      loaded[idx] = true
    end

    return weights, loaded

  end

  --[[Given a word2vec embedings file name and dictionary, outputs weights. Some portions courtesy of Courtesy of https://github.com/rotmanmi/word2vec.torch]]
  local function loadWord2vec(filename, dict)

    local loaded = tds.Hash()
    local dictSize = dict:size()

    local f = torch.DiskFile(filename, "r")

    -- read header
    f:ascii()
    local numWords = f:readInt()
    local embeddingSize = f:readInt()


    local weights = torch.Tensor(dictSize, embeddingSize)

    -- preload constants
    weights, loaded = preloadSpecial (weights, loaded, dict, embeddingSize)

    -- read content
    f:binary()

    print('processing embeddding file')
    for i = 1, numWords do

      if i%opt.report_every == 0 then
         print(i .. ' embedding tokens reviewed. ' .. #loaded .. ' out of ' .. dictSize .. ' dictionary tokens matched.' )
      end

      local word = readStringv2(f)
      local wordEmbedding = f:readFloat(embeddingSize)
      wordEmbedding = torch.FloatTensor(wordEmbedding)

      local idx = locateIdx(word, dict)

      if idx ~= nil then

        local norm = torch.norm(wordEmbedding, 2)

        -- normalize word embedding
        if norm ~= 0 and opt.normalize == true then
          wordEmbedding:div(norm)
        end

        weights[idx] = wordEmbedding
        loaded[idx] = true

      end

      if #loaded == dictSize then
        print('Quitting early. All ' .. dictSize .. ' dictionary tokens matched.')
        break
      end

    -- End File loop
    end

    if #loaded ~= dictSize then
      print('Embedding file fully processed. ' .. #loaded .. ' out of ' .. dictSize .. ' dictionary tokens matched.')
      weights = fillGaps(weights, loaded, dictSize, embeddingSize)
      print('Remaining randomly assigned according to uniform distribution')
    end

    return weights, embeddingSize

  end


  --[[Given a glove embedings file name and dictionary, outputs weights ]]
  local function loadGlove(filename, dict)
    local loaded = tds.Hash()
    local dictSize = dict:size()
    local embeddingSize = nil
    local weights = nil
    local first = true
    local count = 0

    local f = io.open(filename, "r")

    print('processing embeddding file')
    for line in f:lines() do

      count = count + 1
      if count%opt.report_every == 0 then
         print(count .. ' embedding tokens reviewed. ' .. #loaded .. ' out of ' .. dictSize .. ' dictionary tokens matched.' )
      end

      local splitLine = line:split(' ')

      if first == true then
        embeddingSize = #splitLine - 1
        weights = torch.Tensor(dictSize, embeddingSize)

        -- preload constants
        weights, loaded = preloadSpecial (weights, loaded, dict, embeddingSize)
        first = false
      end

      local word = splitLine[1]
      local idx = locateIdx(word, dict)

      if idx ~= nil then

        local wordEmbedding = torch.Tensor(embeddingSize)

        for j = 2, #splitLine do
          wordEmbedding[j - 1] = tonumber(splitLine[j])
        end

        local norm = torch.norm(wordEmbedding, 2)

        -- normalize word embedding
        if norm ~= 0 and opt.normalize == true then
          wordEmbedding:div(norm)
        end

        weights[idx] = wordEmbedding
        loaded[idx] = true

      end

      if #loaded == dictSize then
        print('Quitting early. All ' .. dictSize .. ' dictionary tokens matched.')
        break
      end

    -- End File loop
    end

	f:close()

    if #loaded ~= dictSize then
      print('Embedding file fully processed. ' .. #loaded .. ' out of ' .. dictSize .. ' dictionary tokens matched.')
      weights = fillGaps(weights, loaded, dictSize, embeddingSize)
      print('Remaining randomly assigned according to uniform distribution')
    end

    return weights, embeddingSize

  end



  if embeddingType == "word2vec" then

    return loadWord2vec(embeddingFilename, dictionary)

  elseif embeddingType == "glove" then
    print("Convert GloVe")
    return loadGlove(embeddingFilename, dictionary)

  else

    error('invalid embed type. \'word2vec\' and \'glove\' are the only options.')

  end

end


function getDictName(filePath)
  local dn = string.split(filePath, "/")
  local i, _ = string.find(dn[#dn],".dict")
  return string.sub(dn[#dn],1,i-1)
end

function getEmbedFileName(embedFile)
  local dn = string.split(embedFile, "/")
  local i, _ = string.find(dn[#dn],".txt")
  return string.sub(dn[#dn],1,i-1)
end

local function main()


  --onmt.utils.Opt.init(opt, {"save_data"})



  local timer = torch.Timer()

  assert(path.exists(opt.dict_file), 'dictionary file \'' .. opt.dict_file .. '\' does not exist.')

  --load dictionary
  --local dict = onmt.utils.Dict.new(opt.dict_file)
  local dict = Dict.new(opt.dict_file)

  local embedFile = opt.embed_file
  local embedType = opt.embed_type

  assert(path.exists(embedFile), 'embeddings file \'' .. opt.embed_file .. '\' does not exist.')
  local weights, embeddingSize = loadEmbeddings(embedFile, embedType, dict)

  local outFile = opt.out_folder .. getDictName(opt.dict_file) .. '-' .. getEmbedFileName(opt.embed_file)  .. 'embeddings.t7'
  print('saving weights: ' .. outFile)
  torch.save(outFile, weights)

  print(string.format('completed in %0.3f seconds. ',timer:time().real) .. ' embedding vector size is: ' .. embeddingSize )


end

main()