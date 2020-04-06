# Efficient Time-series Forecasting using Neural Network and Opposition-based Coral Reefs Optimization

[![GitHub release](https://img.shields.io/badge/release-2.0.0-yellow.svg)]()
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3742045.svg)](https://doi.org/10.5281/zenodo.3742045)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

## Dear friends and followers
* I updated to repository to the newest version (version 2.0.0), which is very easy to read and reproduce. 
* All of our optimizers are (meta-heuristics) now deleted and taken the new one from my newest library: 

        https://pypi.org/project/mealpy/
    
* If you use my code or library in your project, I would appreciate the cites:
    * Nguyen, T., Nguyen, T., Nguyen, B. M., & Nguyen, G. (2019). Efficient Time-Series Forecasting Using Neural Network and Opposition-Based Coral Reefs Optimization. International Journal of Computational Intelligence Systems, 12(2), 1144-1161.
	
	* Nguyen, T., Nguyen, B. M., & Nguyen, G. (2019, April). Building Resource Auto-scaler with Functional-Link Neural Network and Adaptive Bacterial Foraging Optimization. In International Conference on Theory and Applications of Models of Computation (pp. 501-517). Springer, Cham.
	
    * Nguyen, T., Tran, N., Nguyen, B. M., & Nguyen, G. (2018, November). A Resource Usage Prediction System Using Functional-Link and Genetic Algorithm Neural Network for Multivariate Cloud Metrics. In 2018 IEEE 11th Conference on Service-Oriented Computing and Applications (SOCA) (pp. 49-56). IEEE.

* If you want to know more about code, or want a pdf of both above paper, contact me: nguyenthieu2102@gmail.com

    
## How to read my repository
1. data: include formatted data
2. utils: Helped functions such as IO, Draw, Math, Settings (for all model and parameters), Preprocessing...
3. paper: include 2 main folders: 
    * results: forecasting results of all models (3 folders inside) 
        * final: final forecasting results (runs on server)
        * stability: final stability results(runs on server)
4. model: (2 folders) 
    * root: (want to understand the code, read this classes first)
        * root_base.py: root for all models (traditional, hybrid and variants...) 
        * traditional: root for all traditional models (inherit: root_base)
        * hybrid: root for all hybrid models (inherit: root_base)
    * main: (final models)
        * this classes will use those optimizer above and those root (traditional, hybrid) above 
        * the running files (outside with the orginial folder: cro_mlnn_script.py, ...) will call this classes
        * the traditional models will use single file such as: traditional_ffnn, traditional_rnn,...
        * the hybrid models will use 2 files, example: hybrid_ffnn.py and GA.py (optimizer files)

    
## Notes

1. To improve the speed of Pycharm when opening (because Pycharm will indexing when opening), you should right click to 
paper and data folder => Mark Directory As  => Excluded

2. How to run models? 
```code 
1. Before runs the models, make sure you clone this repository in your laptop:
    https://github.com/chasebk/code_ocro_mlnn

2. Then open it in your editor like Pycharm or Spider...

3. Now you need to create python environment using conda (assumpted that you have already had it). Open terminal
    conda  your_environment_name  create -f  envs/env.yml           (go to the root project folder and create new environment from my file in: envs/env.yml)

4. Now you can activate your environment and run the models
    conda activate your_environment_name        # First, activate your environment to get the needed libraries.
    python model_name

For example:
    conda ai_env create -f envs/env.yml
    conda activate ai_env
    python lstm1hl_script.py

5. My model name:

    1. MLNN (1 HL) 	=> mlnn1hl_script.py
    2. RNN (1HL)	=> rnn1hl_script.py
    3. LSTM (1HL)	=> lstm1hl_script.py
    4. GA-MLNN 		=> ga_mlnn_script.py
    5. PSO-MLNN 	=> pso_mlnn_script.py
    6. ABFO-MLNN 	=> abfo_mlnn_script.py
    7. CRO-MLNN 	=> cro_mlnn_script.py
    8. OCRO-MLNN 	=> ocro_mlnn_script.py


6. Multi-output meaning?

    cpu: input model would be cpu, output model would be cpu 
    ram: same as cpu
    multi_cpu : input model would be cpu and ram, output model would be cpu 
    multi_ram : input model would be cpu and ram, output model would be ram
    multi : input model would be cpu and ram, output model would be cpu and ram
```

### How to change model's parameters?

```code 
You can change the model's parameters in file: utils/SettingPaper.py 

For example:

+) For traditional models: MLNN, RNN, LSTM

####: MLNN-1HL
mlnn1hl_paras_final = {
    "sliding": [2, 5, 10],
    "hidden_sizes" : [(10, True) ],
    "activations": [("elu", "elu")],  # 0: elu, 1:relu, 2:tanh, 3:sigmoid
    "learning_rate": [0.0001],
    "epoch": [1000],
    "batch_size": [128],
    "optimizer": ["adam"],   # GradientDescentOptimizer, AdamOptimizer, AdagradOptimizer, AdadeltaOptimizer
    "loss": ["mse"]
}

- If you want to tune the parameters, you can adding more value in each parameters like this:

- sliding: [1, 2, 3, 4] or you want just 1 parameter: [12]
- hidden_sizes: [ (5, True), (10, True), (1000, True) ] or [ (14, True) ]
- activations: [ ("elu", "relu"), ("elu", "tanh") ] or just: [ ("elu", "elu") ]
- learning_rate: [0.1, 0.01, 0.001] or just: [0.1]
....


+ For hybrid models: GA_MLNN, PSO_MLNN, CRO_MLNN, OCRO_MLNN
 
#### : GA-MLNN
ga_mlnn_paras_final = {
    "sliding": [2, 5, 10],
    "hidden_size" : [(10, True) ],
    "activations": [(0, 0)],             # 0: elu, 1:relu, 2:tanh, 3:sigmoid
    "train_valid_rate": [(0.6, 0.4)],

    "epoch": [1000],
    "pop_size": [100],                  # 100 -> 900
    "pc": [0.95],                       # 0.85 -> 0.97
    "pm": [0.025],                      # 0.005 -> 0.10
    "domain_range": [(-1, 1)]           # lower and upper bound
}

- Same as traditional models.
```

### Where is the results folder?
```code 
- Look at the running file, for example: ga_elm_script.py

+) For 1-time runs (Only run 1 time for each model).
    _There are 3 type of results file include: model_name.csv file, model_name.png file and Error-model_name.csv file 
    _model_name.csv included: y_true and y_predict 
    _model_name.png is visualized of: y_true and y_predict (test dataset)
    _Error-model_name.csv included errors of training dataset after epoch. 1st, 2nd column are: MSE errors

=> All 3 type of files above is automatically generated in folder which you can set in SettingPaper

+) For stability runs (Run each model n-time with same parameters).
Because in this test, we don't need to visualize the y_true and y_predict and also don't need to save y_true and y_predict
So I just save n-time running in the same csv files in folder

- Noted:

+ In the training set we use MSE. But for the testing set, we can use so much more error like: R2, MAPE, ...
You can find the function code described it in file: 
    model/root/root_base.py 
        + _save_results__: 
```

* Take a look at project_structure.md file.  Describe how the project was built.



### Publications
* If you see our work is useful, please cite us as follows:
```code
   @article{Nguyen2019,
  title={Efficient Time-Series Forecasting Using Neural Network and Opposition-Based Coral Reefs Optimization},
  author={Thieu Nguyen and Tu Nguyen and Binh Minh Nguyen and Giang Nguyen},
  year={2019},
  journal={International Journal of Computational Intelligence Systems},
  volume={12},
  issue={2},
  pages={1144-1161},
  issn={1875-6883},
  url={https://doi.org/10.2991/ijcis.d.190930.003},
  doi={https://doi.org/10.2991/ijcis.d.190930.003}
}
```

* Link full PDF:
	=> [PDF](https://download.atlantis-press.com/article/125921354.pdf)
