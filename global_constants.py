# Data
DIM = 2 # the dimensionality of the data
NR_DETECTORS = 5
EVENTS = 50000
MAX_NR_TRACKS = 3
NOISE_STD = 0.1
HITS_DATA_PATH = 'data/hits_dataframe_dataset1.csv'
TRACKS_DATA_PATH = 'data/tracks_dataframe_dataset1.csv'
TEST_SPLIT = 0.2 # amount from the full dataset
TRAIN_SPLIT = 0.9 # amount from the dataset left over after the test split is set aside

# Training
NUM_EPOCHS = 10
BATCH_SIZE = 256
TEST_BATCH_SIZE = 1
EARLY_STOPPING = 30
PAD_TOKEN = 101 # just a random number outside of the range (-pi,pi) and not possible to be a label number so it does not confuse the model
MAX_LEN_DATA = MAX_NR_TRACKS * NR_DETECTORS
DISABLE_TQDM = False

# Models
# Simple regressor model (Hits -> Trajectories) parameters
OUTPUT_SIZE_REGRESS = 3 # the fixed number of tracks
HIDDEN_SIZE_REGRESS = 32
DROPOUT_REGRESS = 0.2
LEARNING_RATE_REGRESS = 1e-2

# Direct transformer (Hits -> Trajectories) parameters
TR_D_MODEL = 64
TR_N_HEAD = 8
TR_DIM_FEEDFORWARD = 64
TR_NUM_ENCODER_LAYERS = 8
TR_LEARNING_RATE = 1e-3
TR_DROPOUT = 0.15

# Classifying transformer (Hits -> Clusters) parameters
CL_D_MODEL = 32
CL_N_HEAD = 4
CL_DIM_FEEDFORWARD = 64
CL_NUM_ENCODER_LAYERS = 5
CL_LEARNING_RATE = 1e-2
CL_DROPOUT = 0.1

# Regressor RNN (Clusters -> Trajectories) parameters
HIDDEN_SIZE_RNN = 64
OUTPUT_SIZE_RNN = 1
MAX_CLUSTER_SIZE = 5
RNN_LEARNING_RATE = 1e-3