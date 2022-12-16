env_id = "gym_mc:mc-v0"
MAX_EPISODES = 400
MAX_STEPS = 1000
BATCH_SIZE = 64
N = 500
MEASUREMENT_INTERVAL = 40
N_RUNS = 1
LEARNING_RATE = 1e-3

LOAD_FROM_FILE = False

TSNE_COLOR = 'actions' # or 'actions', 'velocity_sign', 'velocity', 'position_and_velocity', 'max_action_values'

PROCESS_TSNE_POSITIVES = False

RANDOM_SEED = 0 # set to None for random runs


LOOK_FOR_CONTINUALLY_INCREASING_REWARD = False