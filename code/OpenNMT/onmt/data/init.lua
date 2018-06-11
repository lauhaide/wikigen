local data = {}

data.Dataset = require('onmt.data.Dataset')
data.Batch = require('onmt.data.Batch')

--data.BoxDataset = require('onmt.data.BoxDataset')
data.BoxBatch = require('onmt.data.BoxBatch')
data.BoxDataset2 = require('onmt.data.BoxDataset2')
--data.BoxBatch2 = require('onmt.data.BoxBatch2')
data.BoxBatch3 = require('onmt.data.BoxBatch3')
data.BoxSwitchBatch = require('onmt.data.BoxSwitchBatch')

--My classes
package.path = '../data2text-align/s2sattn_dta/?.lua;' .. package.path
require 'edData'
data.WikiDataset2 = require('onmt.data.WikiDataset2')
data.WikiDataBatch3 = require('onmt.data.WikiDataBatch3')
data.WikiHierarchicalBatch = require('onmt.data.WikiHierarchicalBatch')
data.WikiDummyBatch3 = require('onmt.data.WikiDummyBatch3')

return data
