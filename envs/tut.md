## How to install and use environment
```code 

1) Update conda 
    conda --version     
    conda update conda 

2) Create new environment in conda
    
    + conda create -n myenv python=3.7.4

    + conda env create -f environment.yml (the first line in that file is the name of the environment)

3) Activate, check and deactivate environment
    conda activate myenv       

    conda list          (or)
    conda env list 
    conda info --envs    

    conda deactivate           
    
4) Check package inside the environment 
    conda list -n myenv         (if myenv hasn't activated)
    conda list                  (if myenv already activated)
    
5) Export to .yml for other usage.  
    source activate myenv                  (access to environment)
    conda env export > environment.yml     

6) Delete environment 
    conda remove --name myenv --all     (or)
    conda env remove --name myenv   
    
    conda info --envs   (kiểm tra xem đã xóa chưa)
```

## Link libraries needed
https://bigdata-madesimple.com/top-20-python-libraries-for-data-science/
