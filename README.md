# TrustBench
A Benchmark for Trustworthiness Evaluation of Machine Learning Models 

## Installation
TrustBench is implemented in Python 3.10. To install the required packages, run:

```shell
#Optional: Create a virtual environment
$ python3.10 -m venv env
$ source env/bin/activate
$ pip install -r requirements.txt
$ export TRUSTBENCH_PATH=~/.trustbench
$ mkdir $TRUSTBENCH_PATH
$ cp -r config $TRUSTBENCH_PATH
```

## Configuration

Kaggle requires an API configuration to use. 
1. Create the Kaggle API key under your profile Settings. 
2. Save the `kaggle.json` file under the `.kaggle` folder in your home directory.
   1. Linux - `~/.kaggle/kaggle.json`;
   2. Windows - `C:\Users\<Windows-username>\.kaggle\kaggle.json`.

## Usage
```shell
usage: python3.10 -m trustbench.main
```
