# Efficient Time-series Forecasting using Neural Network and Opposition-based Coral Reefs Optimization

## Publications
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

## Contact
* If you want to know more about code, or want a pdf of both above paper, contact me: nguyenthieu2102@gmail.com

* Take a look at this repos, the simplify code using python (numpy) for all algorithms above. (without neural networks)
	
	* https://github.com/thieunguyen5991/metaheuristics

## License

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
  
