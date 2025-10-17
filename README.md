# SMC-Dynamic-Loss-Reweighting-for-Once-for-All-Dataset
## Quick Start

### Installation

Download repo:

```
git clone https://github.com/HengShu-Iscas/SMC-Dynamic-Loss-Reweighting-for-Once-for-All-Dataset.git
cd SMC-Dynamic-Loss-Reweighting-for-Once-for-All-Dataset
```

Create pytorch environment:

```
conda env create -f environment.yaml
conda activate smc
```

### Condensing

MDC Condense:

```
python condense_reg.py --reproduce -d [DATASET] -f [FACTOR] --ipc [IPC] --adaptive_reg True --use_mmd --gamma_schedule --gamma_start [START_GAMMA]--gamma_end [END_GAMMA]

# Example on CIFAR-10, IPC10
python condense_reg.py --reproduce -d cifar10 -f 2 --ipc 10 --adaptive_reg True --use_mmd --gamma_schedule --gamma_start 1 --gamma_end 0.2 
```

### Testing

To evaluate a condensed dataset, run:

```
python test.py --reproduce -d [DATASET] -f [FACTOR] --ipc [IPC] --test_type [CHOICES] --test_data_dir [PATH_TO_CONDENSED_DATA_DIR] --ipcy [IPCY]

# Example of evaluating the performance of IPC5 from CIFAR-10, IPC10 (repeating 3 times).
python test.py --reproduce -d cifar10 -f 2 --ipc 10 --test_type cx_cy --test_data_dir ./path_to_ipc10_data --ipcy 4 --repeat 3

```


