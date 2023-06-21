# Flexible Dual-Branch Siamese Network: Learning Location Quality Estimation and Regression Distribution for Visual Tracking
IEEE Transactions on Computational Social Systems

## 1.result
<table>
    <tr>
        <td colspan="2" align=center> Dataset</td>
        <td align=center>SiamFDB</td>
    </tr>
    <tr>
        <td rowspan="2" align=center>OTB100</td>
        <td>Success</td>
        <td>70.4</td>
    </tr>
    <tr>
        <td>Precision</td>
        <td>91.9</td>
    </tr>
    <tr>
        <td rowspan="2" align=center>UAV123</td>
        <td>Success</td>
        <td>64.9</td>
    </tr>
    <tr>
        <td>Precision</td>
        <td>84.0</td>
    </tr>
    <tr>
        <td rowspan="3" align=center>LaSOT</td>
        <td>Success</td>
        <td>52.7</td>
    </tr>
    <tr>
        <td>Norm precision</td>
        <td>60.9</td>
    </tr>
    <tr>
        <td>Precision</td>
        <td>54.0</td>
    </tr>
    <tr>
        <td rowspan="3" align=center>GOT10k</td>
        <td>AO</td>
        <td>63.5</td>
    </tr>
    <tr>
        <td>SR0.5</td>
        <td>73.8</td>
    </tr>
    <tr>
        <td>SR0.75</td>
        <td>51.0</td>
    </tr>
        <tr>
        <td rowspan="3" align=center>VOT2019</td>
        <td>EAO</td>
        <td>31.7</td>
    </tr>
    <tr>
        <td>Accuracy</td>
        <td>60.3</td>
    </tr>
    <tr>
        <td>Robustness</td>
        <td>42.6</td>
    </tr>
</table>

## 2. Environment setup
This code has been tested on Ubuntu 18.04, Python 3.7, Pytorch 1.7.1, CUDA 11.
Please install related libraries before running this code: 
```bash
pip install -r requirements.txt
```

### Download the pretrained model:  
The model has been trained for testing and validation.[model](https://pan.baidu.com/s/1aVYFDC-11eD-RDK85BSMSQ) code: xyqt

Download testing datasets and put them into `test_dataset` directory. If you want to test the tracker on a new dataset, please refer to [pysot-toolkit](https://github.com/Giveupfree/SOTDrawRect) to set test_dataset.


### Download pretrained backbones
Download pretrained backbones from [google driver](https://drive.google.com/drive/folders/1DuXVWVYIeynAcvt9uxtkuleV6bs6e3T9) or [BaiduYun](https://pan.baidu.com/s/1IfZoxZNynPdY2UJ_--ZG2w) (code: 7n7d) and put them into `pretrained_models` directory.


## 3.train
To train the SiamFDB model, run train.py with the desired configs:
For OTB and VOT Benchmark.
```bash
cd /path/to/SiamFDB
export PYTHONPATH=./:$PYTHONPATH
python tools/train.py --cfg ./experiments/SiamFDB_r50/configOTBVOT.yaml
```
For UAV Benchmark
```bash
cd /path/to/SiamFDB
export PYTHONPATH=./:$PYTHONPATH
python tools/train.py --cfg ./experiments/SiamFDB_r50/config.yaml
```
For GOT10k Benchmark
```bash
cd /path/to/SiamFDB
export PYTHONPATH=./:$PYTHONPATH
python tools/train.py --cfg ./experiments/SiamFDB_r50_got10k/config.yaml
```
For LaSOT Benchmark
```bash
cd /path/to/SiamFDB
export PYTHONPATH=./:$PYTHONPATH
python tools/train.py --cfg ./experiments/SiamFDB_r50_lasot/config.yaml
```

## 4.test
```bash
cd /path/to/SiamFDB
export PYTHONPATH=./:$PYTHONPATH
python tools/test.py --config ./experiments/SiamFDB_r50/config.yaml --dataset UAV123 --snapshot ./models/UAV123.pth
```

## 5.eval
please refer to [pysot-toolkit](https://github.com/Giveupfree/SOTDrawRect)

## 6.tune
```bash
cd /path/to/SiamFDB
export PYTHONPATH=./:$PYTHONPATH
python tools/tune.py                                \
	--dataset_root  /path/to/dataset/root            \ # dataset path
	--dataset UAV123                                \ # dataset name(OTB100, GOT10k, LaSOT, UAV123, VOT2016, VOT2018, VOT2019)
	--snapshot ./models/UAV123.pth           \ # tracker_name
	--config ./experiments/SiamFDB_r50/config.yaml   \ # config file
```

## 7.Cite
If you use SiamFDB in your work please cite our paper:
> @ARTICLE{10034430,  
  author={Hu, Shuo and Zhou, Sien and Lu, Jinbo and Yu, Hui},  
  journal={IEEE Transactions on Computational Social Systems},  
  title={Flexible Dual-Branch Siamese Network: Learning Location Quality Estimation and Regression Distribution for Visual Tracking},  
  year={2023},  
  volume={},  
  number={},  
  pages={1-9},  
  doi={10.1109/TCSS.2023.3235649}  
}
