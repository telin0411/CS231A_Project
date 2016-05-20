require 'torch'
require 'paths'
require 'cudnn'
require 'cunn'
require 'image'
local t = require 'data/transforms'
local utils = require 'utils'


-------------------------------------------------------------------------------
-- params
-------------------------------------------------------------------------------

local data_info = 'data/coco.json'
local data_dir = '/scr/r6/karpathy/coco'
local model_path = 'models/resnet-101.t7'

-------------------------------------------------------------------------------
-- utils
-------------------------------------------------------------------------------
local function loadImage(path)
  local ok, input = pcall(function()
    return image.load(path, 3, 'float')
  end)
  -- Sometimes image.load fails because the file extension does not match the
  -- image format. In that case, use image.decompress on a ByteTensor.
  if not ok then
    local f = io.open(path, 'r')
    assert(f, 'Error reading: ' .. tostring(path))
    local data = f:read('*a')
    f:close()
    local b = torch.ByteTensor(string.len(data))
    ffi.copy(b:data(), data, b:size(1))
    input = image.decompress(b, 3, 'float')
  end
  return input
end

-------------------------------------------------------------------------------
-- set up the model
-------------------------------------------------------------------------------

print('loading model ' .. model_path)
local model = torch.load(model_path)
-- Remove the fully connected layer at the top
assert(torch.type(model:get(#model.modules)) == 'nn.Linear')
model:remove(#model.modules)
model:evaluate()
-- The model was trained with this input normalization
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}
local transform = t.Compose{
   t.Scale(256),
   t.ColorNormalize(meanstd),
   t.CenterCrop(224),
}

-------------------------------------------------------------------------------
-- process the files
-------------------------------------------------------------------------------
print('reading json blob ' .. data_info)
local blob = utils.read_json(data_info)

local features = {}
for i, split in ipairs{'train', 'val'} do
   print('processing split ', split)
   local image_info = blob[split]
   local n = #image_info
   local feat_tensor = nil
   for k, j in ipairs(image_info) do
      local image = loadImage(paths.concat(data_dir, j['file_path'])) -- RGB float tensor of values 0..1
      image = transform(image) -- scale, normalize, crop
      image = image:view(1, table.unpack(image:size():totable())) -- view as mini-batch of size 1
      image = image:cuda() -- ship to GPU
      local output = model:forward(image):float():squeeze(1) -- get features and flatten
      if not feat_tensor then
         -- lazy init
         local feat_dim = output:size(1)
         print('determined feature dimension to be ', feat_dim)
         feat_tensor = torch.FloatTensor(n, feat_dim)
      end
      feat_tensor[k]:copy(output)

      --if k-1 % 1000 == 0 then
      print(string.format('%d/%d...', k, n))
      --end
   end
   features[split] = feat_tensor
end

torch.save('features.t7', features)
print('saved features to features.t7')
