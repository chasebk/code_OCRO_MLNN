# Efficient Time-series Forecasting using Neural Network and Opposition-based Coral Reefs Optimization

## Publications
* If you see my code and data useful and use it, please cites us here
    * Th. Nguyen, T. Nguyen, B.M. Nguyen, G. Nguyen: Efficient Time-series Forecasting using Neural Network and Opposition-based Coral Reefs Optimization. International Journal of Computational Intelligence Systems, Volume, Issue, pp., ISSN 1875-6891, DOI. Atlantis Press, 2019. SCIE. CC BY 4.0. (accepted)
    
## How to read my repository
1. data: include formatted data
2. utils: Helped functions such as IO, Draw, Math, Settings (for all model and parameters), Preprocessing...
3. paper: include 2 main folders: 
    * results: forecasting results of all models (3 folders inside) 
        * final: final forecasting results (runs on server)
        * stability: final stability results(runs on server)
4. model: (4 folders) 
    * root: (want to understand the code, read this classes first)
        * root_base.py: root for all models (traditional, hybrid and variants...) 
        * root_algo.py: root for all optimization algorithms
        * traditional: root for all traditional models (inherit: root_base)
        * hybrid: root for all hybrid models (inherit: root_base)
    * optimizer: (this classes inherit: root_algo.py)
        * evolutionary: include algorithms related to evolution algorithm such as GA, DE,..
        * swarm: include algorithms related to swarm optimization such as PSO, CSO, BFO, ...
    * main: (final models)
        * this classes will use those optimizer above and those root (traditional, hybrid) above 
        * the running files (outside with the orginial folder: cro_mlnn_script.py, ...) will call this classes
        * the traditional models will use single file such as: traditional_ffnn, traditional_rnn,...
        * the hybrid models will use 2 files, example: hybrid_ffnn.py and GA.py (optimizer files)

    
## Notes
1. To improve the speed of Pycharm when opening (because Pycharm will indexing when opening), you should right click to 
paper and data folder => Mark Directory As  => Excluded

2. When runs models, you should copy the running files to the original folder (prediction_flnn folder)

3. Make sure you active the environment before run the running files 
* For terminal on linux
```code
    source activate environment_name 
    python running_file.py (python cro_mlnn_script.py)
```
4. In paper/results/final model includes folder's name represent the data such as 
```code
cpu: input model would be cpu, output model would be cpu 
ram: same as cpu
multi_cpu : input model would be cpu and ram, output model would be cpu 
multi_ram : input model would be cpu and ram, output model would be ram
multi : input model would be cpu and ram, output model would be cpu and ram
```
5. Take a look at project_structure.md file.  Describe how the project was built.

## Model
```code
1. MLNN (1 HL) 	=> mlnn1hl_script.py
2. RNN (1HL)		=> rnn1hl_script.py
3. LSTM (1HL)	=> lstm1hl_script.py
4. GA-MLNN 		=> ga_mlnn_script.py
5. PSO-MLNN 	=> pso_mlnn_script.py
6. ABFO-MLNN 	=> abfo_mlnn_script.py
7. CRO-MLNN 	=> cro_mlnn_script.py
8. OCRO-MLNN 	=> ocro_mlnn_script.py
```

## Notes
* If you want to know more about code, or want a pdf of both above paper, contact me: nguyenthieu2102@gmail.com

* Take a look at this repos, the simplify code using python (numpy) for all algorithms above. (without neural networks)
	
	* https://github.com/thieunguyen5991/metaheuristics

## License

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
  
