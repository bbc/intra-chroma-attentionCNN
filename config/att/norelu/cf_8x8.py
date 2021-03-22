experiment_name = "base1"
output_path = "/work/marcb/Intra_Chroma/experiments/att_norelu"
data_path = "/work/marcb/data/DIV2K/0.60-1000-1"

# Model parameters
model = 'norelu'             # [relu, norelu]
block_shape = '8x8'          # block shape of the model
bb1, bb2 = 32, 64            # channels of the 2 boundary branch layers
lb1, lb2 = 64, 64            # channels of the 2 luma convolutional branch layers
tb = 64                      # channels of the predicted head
att_h = 16                   # hidden dimension of the attention module.
ext_bound = True             # boundary of 2N + 1 samples (True) or N + 1 samples (False)
temperature = 0.5            # temperature of the softmax operation

# Training parameters
epochs = 90                  # number of epochs
batch_size = 16              # batch size
validate = True              # perform validation loop or not
shuffle = True               # shuffle dataset or not
use_multiprocessing = False  # use multiprocessing in the data loader

# Optimizer
lr = 0.00001                 # learning rate
beta = 0.9                   # beta parameter in the learning rate

# Early stop
es_patience = 15             # number of epochs for early stop

