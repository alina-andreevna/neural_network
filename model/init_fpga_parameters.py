from init_network_parameters import HIDDEN_LAYERS_COUNT

INPUT_DATA_SIZE = 10

# LINEAR PROPOGARDATION

# Input layer
COEFF_INPUT_W_SIZE = 10
COEFF_INPUT_B_SIZE = 10

OUTPUT_AFTER_INPUT_LAYER_DATA_SIZE = 12

# Hidden layers
COEFF_HIDDEN_W_SIZE = [10, 10, 10]
COEFF_HIDDEN_B_SIZE = [10, 10, 10]

OUTPUT_AFTER_HIDDEN_LAYER_DATA_SIZE = [OUTPUT_AFTER_INPUT_LAYER_DATA_SIZE + 2,
                                    OUTPUT_AFTER_INPUT_LAYER_DATA_SIZE + 4,
                                    OUTPUT_AFTER_INPUT_LAYER_DATA_SIZE + 6]

# Output layer
COEFF_OUTPUT_W_SIZE = 10
COEFF_OUTPUT_B_SIZE = 10

OUTPUT_LAYER_DATA_SIZE = OUTPUT_AFTER_HIDDEN_LAYER_DATA_SIZE[-1] + 2

# Cost
LOG_COST_DATA_SIZE = OUTPUT_LAYER_DATA_SIZE + 1
COST_DATA_SIZE = LOG_COST_DATA_SIZE + 2

# Assertionss
assert(len(OUTPUT_AFTER_HIDDEN_LAYER_DATA_SIZE) == HIDDEN_LAYERS_COUNT)

# BACKWARD PROPOGARDATION

OUTPUT_AFTER_OUTPUT_BACKLAYER_DATA_SIZE = OUTPUT_LAYER_DATA_SIZE + 2

OUTPUT_AFTER_HIDDEN_BACKLAYER_DATA_SIZE = [OUTPUT_AFTER_OUTPUT_BACKLAYER_DATA_SIZE + 2,
                                    OUTPUT_AFTER_OUTPUT_BACKLAYER_DATA_SIZE + 4,
                                    OUTPUT_AFTER_OUTPUT_BACKLAYER_DATA_SIZE + 6]

OUTPUT_AFTER_INPUT_BACKLAYER_DATA_SIZE = OUTPUT_AFTER_HIDDEN_BACKLAYER_DATA_SIZE[-1] + 2

# Assertionss
assert(len(OUTPUT_AFTER_HIDDEN_BACKLAYER_DATA_SIZE) == HIDDEN_LAYERS_COUNT)
