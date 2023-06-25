# data
DIM = 2 # the dimensionality of the data
NR_DETECTORS = 5
EVENTS = 5000
MAX_NR_TRACKS = 20
NOISE_STD = 0.1
HITS_DATA_PATH = 'hits_dataframe.csv'
TRACKS_DATA_PATH = 'tracks_dataframe.csv'
TEST_SPLIT = 0.2
TRAIN_SPLIT = 0.9

# transformer
NUM_EPOCHS = 1
BATCH_SIZE = 32
TEST_BATCH_SIZE = 1
INPUT_SIZE = 3
OUTPUT_SIZE = 20
D_MODEL = 64
N_HEAD = 4
DIM_FEEDFORWARD = 1
NUM_ENCODER_LAYERS = 4
LEARNING_RATE = 1e-4
BETA1 = 0.9
BETA2 = 0.98
EPS = 1e-9
EARLY_STOPPING = 30
PAD_TOKEN = 101 # just a random number outside of the range (-pi,pi) and not possible to be a label number so it does not confuse the model
PAD_LEN_DATA = MAX_NR_TRACKS * NR_DETECTORS