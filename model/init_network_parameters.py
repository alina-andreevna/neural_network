# Input description
INPUT_LAYER_SIZE = 5
INPUT_ACTIVATION = 'relu'

# Hidden description
HIDDEN_LAYERS_COUNT = 3
HIDDEN_LAYERS_SIZE = [4,3,2]
HIDDER_LAYERS_ACTIVATIONS = ['relu', 'relu', 'relu']

# Output description
OUTPUT_LAYER_SIZE = 1
OUTPUT_ACTIVATION = 'sigmoid'

# Assertions
assert(len(HIDDEN_LAYERS_SIZE) == HIDDEN_LAYERS_COUNT)
assert(len(HIDDER_LAYERS_ACTIVATIONS) == HIDDEN_LAYERS_COUNT)

assert(OUTPUT_LAYER_SIZE == 1)
