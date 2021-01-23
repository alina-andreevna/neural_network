## Multylayer neural network

Customized neurl network based on preceptron for FPGA/ASIC.
Trained for check picture: cat or not cat.

**Languages:**
* model: Python
* src: Verilog
* sim: SystemVerilog

**Features:**
*
* variable counts of inputs, layers, layer sizes, etc.
* variable data_bit_size and coeff_data_size in all levels
* customize activation functions: RELU or Sigmoid;
* in the future: customize optimizer (Adam, SGD, Nesterov, no-optimizer)

#### Model ToDo list
~~1. Choose train/test dataset~~  
2. Create functions for model (*all functions should be in fixed point*)    
~~1.1 Initilization~~  
~~1.2 Forward propogardation~~  
~~1.3 Calculate cost~~  
1.4 Backward propogardation  
1.5 Calculate loss  
1.6 Update weigths  
1.7 Make prediction  
2. Create network model
3. Train and check network model
4. Choose checkpoints for validation
5. Create .txt file with checkpoints
7. Create .txt for initilize network weights
6. Make documentation for model

#### Sources ToDo list
1. Create file with initial settings
2. Choose type for all functions in model: function or module
3. Create all functions and modules
4. Create top level file for model
5. Marked all checkpoints like comments in code
6. Make documentation for sources


#### Verifiction ToDo list (using UVM)
1. Create simple testbench (tb) for stimulate inputs
2. Check absense signal into model in each steps
3. Add checkpoints from model
4. Compare model and sources outputs
5. Recomended to use UVM or Python for high-level verification
