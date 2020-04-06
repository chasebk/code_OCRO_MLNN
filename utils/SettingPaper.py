###### Config for test

SP_RUN_TIMES = 3
SP_LOG_FILENAME = "LOG_MODELS"
SP_PATH_SAVE_BASE = "history/results/"
SP_DRAW = True
SP_PRINT_TRAIN = 2  # 0: nothing, 1 : full detail, 2: short version
SP_PREPROCESSING_METHOD = 0  # 0: sliding window, 1: mean, 2: min-mean-max, 3: min-median-max

SP_DATA_SPLIT_INDEX = (0.8, 0, 0.2)
SP_DATA_SPLIT_INDEX_2 = (0.75, 0.15, 0.15)

### Full avaiable dataset
SP_LOAD_DATA_FROM = "data/formatted/"
SP_DATA_FILENAME = ["it_eu_5m", "it_uk_5m", "worldcup98_5m", "gg_cpu", "gg_ram", "gg_multi_cpu", "gg_multi_ram"]
SP_DATA_COLS = [[4], [2], [2], [1], [2], [1, 2], [1, 2]]
SP_DATA_MULTI_OUTPUT = [False, False, False, False, False, False, False]
SP_OUTPUT_INDEX = [None, None, None, None, None, 0, 1]


######################## Paras according to the paper

####: MLNN-1HL
mlnn1hl_paras_final = {
    "sliding": [2, 5, 10],
    "hidden_sizes": [(10, True) ],
    "activations": [("elu", "elu")],
    "learning_rate": [0.0001],
    "epoch": [1000],
    "batch_size": [128],
    "optimizer": ["adam"],   # GradientDescentOptimizer, AdamOptimizer, AdagradOptimizer, AdadeltaOptimizer
    "loss": ["mse"]
}

####: RNN-1HL
rnn1hl_paras_final = {
    "sliding": [2, 5, 10],
    "hidden_sizes": [(10, True)],
    "activations": [("elu", "elu")],
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
    "hidden_sizes": [(10, True)],
    "activations": [("elu", "elu")],
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
    "hidden_sizes": [(10, True)],
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
    "hidden_size": [(10, True)],
    "activations": [("elu", "elu")],

    "epoch": [1000],
    "pop_size": [100],                  # 100 -> 900
    "pc": [0.95],                       # 0.85 -> 0.97
    "pm": [0.025],                      # 0.005 -> 0.10
    "domain_range": [(-1, 1)]           # lower and upper bound
}

#### : DE-MLNN
de_mlnn_paras_final = {
    "sliding": [2, 5, 10],
    "hidden_size": [(10, True)],
    "activations": [("elu", "elu")],

    "epoch": [1000],
    "pop_size": [100],                  # 10 * problem_size
    "Wf": [0.8],                        # Weighting factor
    "Cr": [0.9],                        # Crossover rate
    "domain_range": [(-1, 1)]           # lower and upper bound
}


#### : CRO-MLNN
cro_mlnn_paras_final = {
    "sliding": [2, 5, 10],
    "hidden_size": [(10, True) ],
    "activations": [("elu", "elu")],

    "epoch": [1000],
    "pop_size": [100],
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
    "hidden_size": [(10, True)],
    "activations": [("elu", "elu")],

    "epoch": [1000],
    "pop_size": [100],
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
    "hidden_size": [(10, True)],
    "activations": [("elu", "elu")],

    "epoch": [1000],
    "pop_size": [100],                  # 100 -> 900
    "w_minmax": [(0.4, 0.9)],  # [0-1] -> [0.4-0.9]      Trong luong cua con chim
    "c_minmax": [(1.2, 1.2)],  # [(1.2, 1.2), (0.8, 2.0), (1.6, 0.6)]     # [0-2]   Muc do anh huong cua local va global
    # r1, r2 : random theo tung vong lap
    # delta(t) = 1 (do do: x(sau) = x(truoc) + van_toc
    "domain_range": [(-1, 1)]           # lower and upper bound
}


#### : BFO-MLNN
bfo_mlnn_paras_final = {
    "sliding": [2, 5, 10],
    "hidden_size": [(10, True)],
    "activations": [("elu", "elu")],

    "pop_size": [100],
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
    "hidden_size": [(10, True)],
    "activations": [("elu", "elu")],

    "epoch": [1000],
    "pop_size": [100],               # 100 -> 900
    "Ci": [(0.1, 0.00001)],         # C_s (start), C_e (end)  -=> step size # step size in BFO
    "Ped": [0.25],                  # p_eliminate
    "Ns": [4],                      # swim_length
    "N_minmax": [(3, 40)],          # (Dead threshold value, split threshold value) -> N_adapt, N_split

    "domain_range": [(-1, 1)]  # lower and upper bound
}
