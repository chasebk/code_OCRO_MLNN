###### Config for test

###: Variables
traffic_eu = [
    "data/formatted/",    # pd.readfilepath
    [4],                    # usecols trong pd
    False,                  # multi_output
    None,                   # output_idx
    "eu/",                  # path_save_result
]
traffic_uk = [
    "data/formatted/",    # pd.readfilepath
    [2],                    # usecols trong pd
    False,                  # multi_output
    None,                   # output_idx
    "uk/",                  # path_save_result
]

worldcup = [
    "data/formatted/",    # pd.readfilepath
    [2],            # usecols trong pd
    False,          # multi_output
    None,           # output_idx
    "wc/",    # path_save_result
]

ggtrace_cpu = [
    "data/formatted/",    # pd.readfilepath
    [1],         # usecols trong pd
    False,          # multi_output
    None,              # output_idx
    "cpu/",     # path_save_result
]

ggtrace_ram = [
    "data/formatted/",    # pd.readfilepath
    [2],             # usecols trong pd
    False,              # multi_output
    None,                  # output_idx
    "ram/",       # path_save_result
]

ggtrace_multi_cpu = [
    "data/formatted/",    # pd.readfilepath
    [1, 2],         # usecols trong pd
    False,          # multi_output
    0,              # output_idx
    "multi_cpu/",     # path_save_result
]

ggtrace_multi_ram = [
    "data/formatted/",    # pd.readfilepath
    [1, 2],             # usecols trong pd
    False,              # multi_output
    1,                  # output_idx
    "multi_ram/",       # path_save_result
]


######################## Paras according to the paper

####: MLNN-1HL
mlnn1hl_paras_final = {
    "sliding": [2, 5, 10],
    "hidden_sizes" : [[5] ],
    "activations": [("elu", "elu")],  # 0: elu, 1:relu, 2:tanh, 3:sigmoid
    "learning_rate": [0.0001],
    "epoch": [5000],
    "batch_size": [128],
    "optimizer": ["adam"],   # GradientDescentOptimizer, AdamOptimizer, AdagradOptimizer, AdadeltaOptimizer
    "loss": ["mse"]
}

####: RNN-1HL
rnn1hl_paras_final = {
    "sliding": [2, 5, 10],
    "hidden_sizes" : [[5] ],
    "activations": [("elu", "elu")],  # 0: elu, 1:relu, 2:tanh, 3:sigmoid
    "learning_rate": [0.0001],
    "epoch": [1000],
    "batch_size": [128],
    "optimizer": ["adam"],   # GradientDescentOptimizer, AdamOptimizer, AdagradOptimizer, AdadeltaOptimizer
    "loss": ["mse"],
    "dropouts": [[0.2]]
}

####: LSTM-1HL
lstm1hl_paras_final = {
    "sliding": [2, 5, 10],
    "hidden_sizes" : [[5] ],
    "activations": [("elu", "elu")],  # 0: elu, 1:relu, 2:tanh, 3:sigmoid
    "learning_rate": [0.0001],
    "epoch": [1000],
    "batch_size": [128],
    "optimizer": ["adam"],   # GradientDescentOptimizer, AdamOptimizer, AdagradOptimizer, AdadeltaOptimizer
    "loss": ["mse"],
    "dropouts": [[0.2]]
}

####: GRU-1HL
gru1hl_paras_final = {
    "sliding": [2, 5, 10],
    "hidden_sizes" : [[5] ],
    "activations": [("elu", "elu")],  # 0: elu, 1:relu, 2:tanh, 3:sigmoid
    "learning_rate": [0.0001],
    "epoch": [1000],
    "batch_size": [128],
    "optimizer": ["adam"],   # GradientDescentOptimizer, AdamOptimizer, AdagradOptimizer, AdadeltaOptimizer
    "loss": ["mse"],
    "dropouts": [[0.2]]
}


# ========================= MLNN ==============================

#### : GA-MLNN
ga_mlnn_paras_final = {
    "sliding": [2, 5, 10],
    "hidden_size" : [5 ],
    "activations": [(0, 0)],             # 0: elu, 1:relu, 2:tanh, 3:sigmoid
    "train_valid_rate": [(0.6, 0.4)],

    "epoch": [700],
    "pop_size": [200],                  # 100 -> 900
    "pc": [0.95],                       # 0.85 -> 0.97
    "pm": [0.025],                      # 0.005 -> 0.10
    "domain_range": [(-1, 1)]           # lower and upper bound
}

#### : DE-MLNN
de_mlnn_paras_final = {
    "sliding": [2, 5, 10],
    "hidden_size" : [5 ],
    "activations": [(0, 0)],             # 0: elu, 1:relu, 2:tanh, 3:sigmoid
    "train_valid_rate": [(0.6, 0.4)],

    "epoch": [700],
    "pop_size": [200],                  # 10 * problem_size
    "Wf": [0.8],                        # Weighting factor
    "Cr": [0.9],                        # Crossover rate
    "domain_range": [(-1, 1)]           # lower and upper bound
}


#### : CRO-MLNN
cro_mlnn_paras_final = {
    "sliding": [2, 5, 10],
    "hidden_size" : [5 ],
    "activations": [(0, 0)],             # 0: elu, 1:relu, 2:tanh, 3:sigmoid
    "train_valid_rate": [(0.6, 0.4)],

    "epoch": [700],
    "pop_size": [200],
    "G": [[0.02, 0.2]],
    "GCR": [0.1],
    "po": [0.4],
    "Fb": [0.9],
    "Fa": [0.1],
    "Fd": [0.1],
    "Pd": [0.1],
    "k": [3],
    "domain_range": [(-1, 1)]           # lower and upper bound
}

#### : OCRO-MLNN
ocro_mlnn_paras_final = {
    "sliding": [2, 5, 10],
    "hidden_size" : [5 ],
    "activations": [(0, 0)],             # 0: elu, 1:relu, 2:tanh, 3:sigmoid
    "train_valid_rate": [(0.6, 0.4)],

    "epoch": [700],
    "pop_size": [200],
    "G": [[0.02, 0.2]],
    "GCR": [0.1],
    "po": [0.4],
    "Fb": [0.8],
    "Fa": [0.1],
    "Fd": [0.3],
    "Pd": [0.1],
    "k": [3],
    "domain_range": [(-1, 1)],           # lower and upper bound

    "restart_count": [55],
}



#### : PSO-MLNN
pso_mlnn_paras_final = {
    "sliding": [2, 5, 10],
    "hidden_size" : [5 ],
    "activations": [(0, 0)],             # 0: elu, 1:relu, 2:tanh, 3:sigmoid
    "train_valid_rate": [(0.6, 0.4)],

    "epoch": [700],
    "pop_size": [200],                  # 100 -> 900
    "w_minmax": [(0.4, 0.9)],  # [0-1] -> [0.4-0.9]      Trong luong cua con chim
    "c_minmax": [(1.2, 1.2)],  # [(1.2, 1.2), (0.8, 2.0), (1.6, 0.6)]     # [0-2]   Muc do anh huong cua local va global
    # r1, r2 : random theo tung vong lap
    # delta(t) = 1 (do do: x(sau) = x(truoc) + van_toc
    "domain_range": [(-1, 1)]           # lower and upper bound
}


#### : BFO-MLNN
bfo_mlnn_paras_final = {
    "sliding": [2, 5, 10],
    "hidden_size" : [5 ],
    "activations": [(0, 0)],             # 0: elu, 1:relu, 2:tanh, 3:sigmoid
    "train_valid_rate": [(0.6, 0.4)],

    "pop_size": [50],
    "Ci": [0.01],                       # step_size
    "Ped": [0.25],                      # p_eliminate
    "Ns": [4],                          # swim_length
    "Ned": [6],                                 #  elim_disp_steps
    "Nre": [2],                                 # repro_steps
    "Nc": [30],                                 # chem_steps
    "attract_repel": [(0.1, 0.2, 0.1, 10)],    # [ d_attr, w_attr, h_repel, w_repel ]

    "domain_range": [(-1, 1)]           # lower and upper bound
}

#### : ABFOLS-MLNN
abfols_mlnn_paras_final = {
    "sliding": [2, 5, 10],
    "hidden_size": [5],
    "activations": [(0, 0)],  # 0: elu, 1:relu, 2:tanh, 3:sigmoid
    "train_valid_rate": [(0.6, 0.4)],

    "epoch": [700],
    "pop_size": [200],               # 100 -> 900
    "Ci": [(0.1, 0.00001)],         # C_s (start), C_e (end)  -=> step size # step size in BFO
    "Ped": [0.25],                  # p_eliminate
    "Ns": [4],                      # swim_length
    "N_minmax": [(3, 40)],          # (Dead threshold value, split threshold value) -> N_adapt, N_split

    "domain_range": [(-1, 1)]  # lower and upper bound
}
