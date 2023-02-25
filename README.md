# LHY-ML
The implemented code of the tutorial from Hung-yi Lee's Machine Learning Course

## Environment
```shell
conda create --name=ml python=3.8 -y 
```
```shell
conda activate ml
pip install ipykernel
```
```shell
python -m ipykernel install --user --name ml --display-name "LYH-ML"
```
```shell
jupyter kernelspec uninstall ml
```

### Tensorflow
```shell
conda install -c conda-forge cudatoolkit=12.0 cudnn=8.7.0
```
```shell
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
```
```shell
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```
```shell
pip install tensorflow
```
```shell
sudo ln -s /home/puff/anaconda3/envs/ml/lib/python3.8/site-packages/tensorrt/libnvinfer_plugin.so.8 /home/puff/anaconda3/envs/ml/lib/python3.8/site-packages/tensorrt/libnvinfer_plugin.so.7
sudo ln -s /home/puff/anaconda3/envs/ml/lib/python3.8/site-packages/tensorrt/libnvinfer.so.8 /home/puff/anaconda3/envs/ml/lib/python3.8/site-packages/tensorrt/libnvinfer.so.7
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/puff/anaconda3/lib/
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/puff/anaconda3/envs/ml/lib/python3.8/site-packages/tensorrt


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/puff/anaconda3/envs/ml/lib/python3.8/site-packages/tensorrt/
```