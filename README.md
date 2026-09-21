# PaLoRA

## 1.Requisite

This code is implemented in PyTorch, and we perform the experiments under the following environment settings:

- python = 3.11.4
- torch = 2.0.1
- torchvision = 0.15.2
- timm = 0.6.7

The code has been tested on Linux Platform with a GPU (RTX4080S).

If you see the following error, you may need to install a PyTorch package compatible with your infrastructure.

```
RuntimeError: No HIP GPUs are available or ImportError: libtinfo.so.5: cannot open shared object file: No such file or directory
```

For example if your infrastructure only supports CUDA == 11.1, you may need to install the PyTorch package using CUDA11.1.

```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

If you see the following error, you can resolve it by installing a lower version of timm, such as pip install timm==0.6.7.

```
TypeError: 'PretrainedCfg' object is not subscriptable
```


## 2.Dataset 
 * Create a folder `data/`
 * **CIFAR 100**: should automatically be downloaded
 * **ImageNet-R**: retrieve from [link](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar). After unzipping, place it into `data/` folder
 * **ImageNet-A**: retrieve from [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/ERYi36eg9b1KkfEplgFTW3gBg1otwWwkQPSml0igWBC46A?e=NiTUkL). After unzipping, place it into `data/` folder


## 3.Reproducing
The JSON configuration files in `configs/palora`  contains all the experiments' settings.
- CIFAR100 10TASK:
    ```
    python main.py --config configs/palora/c10_palora.json 
    ```

- ImageNet-R 20TASK:
    ```
    python main.py --config configs/palora/ir20_palora.json
    ```

- ImageNet-A 50TASK:

  ```
  python main.py --config configs/palora/ia50_palora.json
  ```
