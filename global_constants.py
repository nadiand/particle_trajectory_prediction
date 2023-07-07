# data
DIM = 2 # the dimensionality of the data
NR_DETECTORS = 5
EVENTS = 5000
MAX_NR_TRACKS = 3 #20 # 3 for the simple regressor and 20 for the biger challenge
NOISE_STD = 0.1
HITS_DATA_PATH = 'hits_dataframe.csv' #'C:/Users/lenovo/Desktop/uni/MLIPPA/Practical/particle_trajectory_prediction/hits_dataframe.csv'
TRACKS_DATA_PATH = 'tracks_dataframe.csv' #'C:/Users/lenovo/Desktop/uni/MLIPPA/Practical/particle_trajectory_prediction/tracks_dataframe.csv'
TEST_SPLIT = 0.2 # amount from the full dataset
TRAIN_SPLIT = 0.9 # amount from the dataset left over after the test split is set aside

# training
NUM_EPOCHS = 5
BATCH_SIZE = 32
TEST_BATCH_SIZE = 1
EARLY_STOPPING = 30
PAD_TOKEN = 101 # just a random number outside of the range (-pi,pi) and not possible to be a label number so it does not confuse the model
MAX_LEN_DATA = MAX_NR_TRACKS * NR_DETECTORS

INPUT_SIZE = 2
OUTPUT_SIZE = 3#20 # 3 for the transformer (hits -> trajectories)
D_MODEL = 64
N_HEAD = 4
DIM_FEEDFORWARD = 1
NUM_ENCODER_LAYERS = 4
LEARNING_RATE = 1e-4
BETA1 = 0.9
BETA2 = 0.98
EPS = 1e-9