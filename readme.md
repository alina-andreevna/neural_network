## Multylayer neural network
Customized neurl network based on preceptron for FPGA/ASIC

**Features:**
* model language: Python
* src/sim language: Verilog/SystemVerilog
* variable counts of inputs, layers, layer sizes, etc.
* customize activation functions;
* customize optimizer (Adam, SGD, Nesterov, no-optimizer)

#### Model ToDo list
1. Choose train/test dataset
2. Create functions for model (*all functions shold be in fixed point*)
1.1 Initilization
1.2 Forward propogardation
1.3 Backward propogardation
1.4 Calculate loss
1.5 Update weigths
1.6 Make prediction
2. Create network model
3. Train and check network model
4. Choose checkpoints for validation
5. Create .txt file with checkpoints

#### Sources ToDo list
1. Create file with initial settings
2. Choose type for all functions in model: functions or module
3. Create all functions and models
4. Create top level sile for model


#### Verifiction ToDo list (using UVM)
1. Create simple testbench (tb) for stimulate inputs
2. Check absense signal into model in each steps
2. Add checkpoints from model
3. Compare model and sources outputs
