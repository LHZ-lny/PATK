# PATK: Time Series Classification and Prediction Framework

## Project Introduction

PATK is a Transformer-based time series processing framework focused on time series classification and prediction tasks. The framework supports pre-training and fine-tuning modes, and can handle various time series datasets, including physiological signals, sensor data, and meteorological data.

## Main Features

- **Time Series Classification**: Supports multiple classification datasets such as ECG, EMG, Epilepsy, etc.
- **Time Series Prediction**: Supports multiple prediction datasets such as ETT, electricity, weather, etc.
- **Pre-training and Fine-tuning**: Implements a complete workflow for model pre-training and downstream task fine-tuning
- **Multiple Model Support**: Supports various model architectures including Transformer, Mamba, CNN, RNN, etc.
- **Efficient Data Processing**: Includes data loading, preprocessing, and augmentation functionality

## Project Structure

```
PATK/
├── data_load/         # Data loading and preprocessing module
├── dataset/           # Dataset directory
│   ├── Cls/           # Classification datasets
│   └── Pred/          # Prediction datasets
├── model/             # Model definitions
├── trainer/           # Training and testing scripts
├── V2_logs_ETTh1/     # Training logs for ETTh1 dataset
├── V2_logs_SleepEEG/  # Training logs for SleepEEG dataset
├── V2_pred_logs_ETTh1/ # Prediction logs for ETTh1 dataset
├── saved_models_pre_train_ETTh1/    # Pre-trained models for ETTh1
├── saved_models_pre_train_SleepEEG/ # Pre-trained models for SleepEEG
├── main_cls.py        # Main script for classification tasks
├── main_pred.py       # Main script for prediction tasks
├── main_baseline.py   # Baseline model script
├── contrastive_loss_V2.py # Contrastive loss function
└── 实验数据汇总.xlsx   # Experiment results summary
```

## Installation Instructions

### Environment Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA (recommended for GPU acceleration)
- Other dependencies: numpy, einops, matplotlib

### Installation Steps

1. Clone the project to your local machine
   ```bash
   git clone <project address>
   cd PATK
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Classification Task

#### Pre-training

```bash
python main_cls.py --scene pre-train --pre_dataset_name SleepEEG --EPOCH 10 --batch_size 64
```

#### Fine-tuning

```bash
python main_cls.py --scene fine-tune --pre_dataset_name SleepEEG --tune_dataset_name ECG --EPOCH 10
```

### Prediction Task

#### Pre-training

```bash
python main_pred.py --scene pre-train --pre_dataset_name ETTh1 --EPOCH 10 --batch_size 8
```

#### Fine-tuning

```bash
python main_pred.py --scene fine-tune --pre_dataset_name ETTh1 --tune_dataset_name ETTh1 --EPOCH 10
```

## Main Parameter Description

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `--n_layers` | Number of encoding layers | 4 |
| `--n_heads` | Number of multi-head attention heads | 12 |
| `--d_model` | Feature dimension | 128 |
| `--d_ff` | Feed-forward network hidden layer dimension | 512 |
| `--dropout_rate` | Dropout rate | 0.1 |
| `--sampling_num` | Sampling times | 30 |
| `--sampling_length` | Sampling length | 40 |
| `--input_len` | Input sequence length | Classification: 178, Prediction: 336 |
| `--pred_len` | Prediction sequence length | 96 |
| `--dataset` | Task type | Cls or Pred |
| `--pre_dataset_name` | Pre-training dataset name | SleepEEG (classification) or ETTh1 (prediction) |
| `--tune_dataset_name` | Fine-tuning dataset name | ECG (classification) or ETTh1 (prediction) |
| `--scene` | Training scene | pre-train or fine-tune |
| `--EPOCH` | Training epochs | 10 |
| `--batch_size` | Batch size | Classification: 64, Prediction: 8 |
| `--learning_rate` | Learning rate | 0.008 |
| `--model` | Model type | Transformer |

## Model Architecture

### Core Components

1. **RoV_time**: Time domain random sampling module that generates sampling points through change rate distribution, performs slicing and length scaling on time series
2. **SEI_freq**: Frequency domain sampling module that generates sampling points through energy distribution, processes frequency domain signals
3. **Time_transformer**: Transformer-based time series encoder that融合 time domain and frequency domain features
4. **Contrastive Learning**: Uses contrastive loss function for pre-training to enhance model representation capability

### Model Flow

1. Input time series data
2. Perform sampling in time domain and frequency domain through RoV_time and SEI_freq modules respectively
3. Embed and encode the sampling results
4. Extract features through Transformer encoder
5. Fine-tune for classification or prediction tasks

## Supported Datasets

### Classification Datasets

- **ECG**: Electrocardiogram data
- **EMG**: Electromyography data
- **Epilepsy**: Epileptic seizure data
- **FD-A/FD-B**: Fault diagnosis data
- **Gesture**: Gesture recognition data
- **HAR**: Human activity recognition data
- **SleepEEG**: Sleep electroencephalogram data

### Prediction Datasets

- **ETTh1/ETTh2/ETTm1/ETTm2**: Electrical transformer temperature data
- **electricity**: Electricity consumption data
- **illness**: Disease incidence data
- **weather**: Meteorological data

## Experiment Results

Experiment results are summarized in the `实验数据汇总.xlsx` file, including performance metrics of different models on various datasets.

## Example Code

### Classification Task Example

```python
# Pre-training
python main_cls.py --scene pre-train --pre_dataset_name SleepEEG --EPOCH 10

# Fine-tuning
python main_cls.py --scene fine-tune --pre_dataset_name SleepEEG --tune_dataset_name ECG --EPOCH 10
```

### Prediction Task Example

```python
# Pre-training
python main_pred.py --scene pre-train --pre_dataset_name ETTh1 --EPOCH 10

# Fine-tuning
python main_pred.py --scene fine-tune --pre_dataset_name ETTh1 --tune_dataset_name ETTh1 --EPOCH 10
```

## Notes

1. Ensure datasets are correctly placed in the `dataset/` directory
2. Pre-trained models will be saved in the `saved_models_pre_train_<dataset_name>/` directory
3. Training logs will be saved in the `V2_logs_<dataset_name>/` directory
4. For large datasets, GPU training is recommended
5. You can optimize model performance by adjusting `--sampling_num` and `--sampling_length` parameters

## Contribution

Contributions are welcome! Please submit Issues and Pull Requests to improve this project.

## License

This project is licensed under the MIT License.

## Contact

For any questions, please contact the project maintainers.