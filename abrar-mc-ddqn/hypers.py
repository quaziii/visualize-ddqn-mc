env_id = "gym_mc:mc-v0"
MAX_EPISODES = 400
MAX_STEPS = 1000
BATCH_SIZE = 64
N = 500
MEASUREMENT_INTERVAL = 40
N_RUNS = 1
LEARNING_RATE = 1e-4

LOAD_FROM_FILE = False

TSNE_COLOR = 'velocity_sign' # or 'actions', 'velocity_sign', 'velocity', 'position_and_velocity', 'max_action_values'

# EPISODES_TO_ADD_TO_MILESTONES = [60, 61, 62, 63]
EPISODES_TO_ADD_TO_MILESTONES = [60, 61, 62, 63]

PROCESS_TSNE_POSITIVES = False

RANDOM_SEED = 0 # set to None for random runs


LOOK_FOR_CONTINUALLY_INCREASING_REWARD = False