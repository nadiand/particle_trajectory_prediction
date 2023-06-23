# data
NR_DETECTORS = 5
EVENTS = 5000
MAX_NR_TRACKS = 20
NOISE_STD = 0.1
HITS_DATA_PATH = 'hits_dataframe.csv'
TRACKS_DATA_PATH = 'tracks_dataframe.csv'

# transformer
DIM = 2 # the dimensionality of the data
NUM_EPOCHS = 1
BATCH_SIZE = 128
INPUT_SIZE = 3
OUTPUT_SIZE = 3
D_MODEL = 32
N_HEAD = 4
DIM_FEEDFORWARD = 128
NUM_ENCODER_LAYERS = 4
LEARNING_RATE = 1e-4
BETA1 = 0.9
BETA2 = 0.98
EPS = 1e-9
EARLY_STOPPING = 30
PAD_TOKEN = 37 # just a random number outside of the range (-pi,pi) so it does not confuse the model
TEST_BATCH_SIZE = 1