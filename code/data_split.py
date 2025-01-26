from imports import *

original_data = './Data'
data = original_data
 
train = '/train'
test = '/test'
val = '/val'

# os.mkdir(train)
# os.mkdir(test)
#os.mkdir(val)

SEED = 53

splitfolders.ratio(data, output = 'UsableData', seed = SEED, ratio = (.8, .1, .1), group_prefix = None)