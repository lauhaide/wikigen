--
-- Created by IntelliJ IDEA.
-- User: lperez
-- Date: 06/06/17
-- Time: 15:31
-- To change this template use File | Settings | File Templates.
--

local embeddingl = {}


function embeddingl.initRandomEmbeddingMatrix(voc_size, dim)
  print("random embedding initialisation... skip zeros vector...")
  local emb= localize(nn.LookupTableMaskZero(voc_size, dim))
  local l2norm, v
  for i=2,voc_size do
    emb.weight[i] = localize(torch.randn(dim)) ---> gaussian mean=0 and sd=1/sqrt(d)
	--normalize initial weights
	l2norm = math.sqrt(torch.cmul(emb.weight[i],emb.weight[i]):sum())
	v = Tensor(dim):fill(l2norm)
	emb.weight[i]:cdiv(v)
  end
  return emb
end


function embeddingl.initPreTrainedEmbeddingMatrix(voc_size, dim, vectors)
  print("pre-trained embeddings initialisation...")
  local emb= localize(nn.LookupTableMaskZero(voc_size, dim))
  local vecs = load_vectors(vectors)
  emb.weight:sub(2,voc_size+1):copy(vecs)  --LookupTableMaskZero adds an extra dimensions and shifts 1 all indices
  return emb
end


--[[Initialize with pre-trained vectors, then normalise L2]]
function embeddingl.initPreTrainedEmbeddingMatrixL2(voc_size, dim, vectors)
  print("pre-trained embeddings initialisation...")
  local emb= localize(nn.LookupTableMaskZero(voc_size, dim))
  local vecs = load_vectors(vectors)
  emb.weight:sub(2,voc_size+1):copy(vecs)  --LookupTableMaskZero adds an extra dimensions and shifts 1 all indices
  local l2norm, v
  for i=2,voc_size do
    l2norm = math.sqrt(torch.cmul(emb.weight[i],emb.weight[i]):sum())
    v = Tensor(dim):fill(l2norm)
    emb.weight[i]:cdiv(v)
  end
  --print( torch.norm(emb.weight[10],2) ) --debugg
  return emb
end


function embeddingl.getEmbeddingMatrix(word_vec_size, voc_size, preembed_voc, train)
  local emb_data
  if not train then
    emb_data = nn.LookupTableMaskZero(voc_size, word_vec_size)
  else
    if opt.preembed then
      --load pre-trained word vectors
      emb_data = embeddingl.initPreTrainedEmbeddingMatrix(voc_size, word_vec_size, preembed_voc)
    else
      emb_data = embeddingl.initRandomEmbeddingMatrix(voc_size, word_vec_size)
    end
  end
  return emb_data
end


function embeddingl.copyMaskedLookupTables(embTo, embFrom, dictTo, dictFrom)
    local w, idxDTA
    local embeddingDim = embTo:size(2)
    assert(embTo:size(2) == embFrom:size(2), "Embedding layers of different dimension.")
    --Note that the embedding layers have size vocabulary_size + 1 for handling the padding and forward 0's vector
    for idx = 2 , embTo:size(1) do
        w = dictTo[idx-1]
        idxDTA = dictFrom[w]
        --print(w, idxDTA)
        if idxDTA ~= nil then
            local wordEmbedding = embFrom[idxDTA+1]
            local norm = torch.norm(wordEmbedding, 2)
            print(wordEmbedding:size())
            print(wordEmbedding:sum())
            -- normalize word embedding
            if norm ~= 0 and opt.normalize == true then
              wordEmbedding:div(norm)
            end
            print(wordEmbedding:sum())
            os.exit()
            embTo[idx]:copy(wordEmbedding)
        else
            --init missing words
            for e=1, embeddingDim do
                embTo[idx][e] = torch.uniform(-1, 1)
            end
            --todo need to normalise ???
        end
    end
end




return embeddingl