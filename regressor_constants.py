# Simple regressor model (Hit -> Trajectory) parameters
INPUT_SIZE_REGRESS = 2 # for x and y, 2D data only
OUTPUT_SIZE_REGRESS = 3 # the fixed number of tracks
HIDDEN_SIZE_REGRESS = 16
DROPOUT_REGRESS = 0.1
LEARNING_RATE_REGRESS = 1e-4

NUM_EPOCHS = 5
BATCH_SIZE = 32
TEST_BATCH_SIZE = 1
EARLY_STOPPING = 30