import warnings
import torch
import random
import os
import argparse
import trainer.run_cls as runner

cuda = torch.cuda.is_available()

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    ## Parameters of Network
    parser.add_argument('--n_layers', type=int, default=4, help='the number of encoding layers')
    parser.add_argument('--n_heads', type=int, default=12, help='the number of multi-head of self-attention')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of features')
    parser.add_argument('--d_ff', type=int, default=128 * 4, help='the hidden layer')
    parser.add_argument('--d_k', type=int, default=64, help='the dimension of the K matrix')
    parser.add_argument('--d_v', type=int, default=64, help='the dimension of the V matrix')
    parser.add_argument('--dropout_rate', type=int, default=0.1, help='dropout_rate')

    parser.add_argument('--sampling_num', type=int, default=30, help='sampling times')
    parser.add_argument('--sampling_length', type=int, default=40, help='sampling length')
    # parser.add_argument('--freq_sampling_num', type=int, default=10, help='sampling times')
    # parser.add_argument('--freq_sampling_length', type=int, default=60, help='sampling length')
    parser.add_argument('--input_len', type=int, default=178, help='input length')
    parser.add_argument('--pred_len', type=int, default=None, help='prediction length')

    ## Dataset
    parser.add_argument('--dataset', type=str, default='Cls', help='pre-train tasks')
    parser.add_argument('--pre_dataset_name', type=str, default='SleepEEG', help='pretrain dataset name')
    parser.add_argument('--tune_dataset_name', type=str, default='ECG', help='fine tune dataset name')
    parser.add_argument('--data_prepare_files', type=str, default='data_prepare', help='path to custom dataset')
    parser.add_argument('--saved_models_files', type=str, default='saved_models', help='path to custom dataset')

    ## Training
    parser.add_argument('--scene', type=str, default='pre-train', choices=['pre-train', 'fine-tune'], help='target name')
    parser.add_argument('--device', type=bool, default=torch.device('cuda'), help='GPU or CPU')
    parser.add_argument('--EPOCH', type=int, default=10, help='training epoch')
    parser.add_argument('--fine_tune_epochs', type=int, default=10, help='tuning epoch')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--learning_rate', type=int, default=0.008, help='training epoch')
    parser.add_argument('--saved_interval', type=int, default=2, help='model saved interval')
    parser.add_argument('--temp', type=float, default=0.5, help='temperature for loss function')
    parser.add_argument('--model', type=str, default='Transformer', choices=['Mamba', 'Transformer', 'CNN', 'RNN'], help='target name')

    opt = parser.parse_args()
    os.makedirs("./%s_pre_train_%s/" % (opt.saved_models_files, opt.pre_dataset_name), exist_ok=True)

    return opt


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    setup_seed(20240919)
    warnings.filterwarnings("ignore")
    opt = parse_option()
    print('Scene: Pre-train %s dataset, Fine-tune %s dataset' % (opt.pre_dataset_name, opt.tune_dataset_name))

    if opt.scene == 'pre-train':
        trainer = runner.Pre_train(opt)
        trainer.pre_train()

    elif opt.scene == 'fine-tune':
        fine_tuner = runner.Cls_fine_tune(opt)
        fine_tuner.fine_tune_classification(opt.EPOCH, opt.tune_dataset_name)

    else:
        raise TypeError('Please give the correct pre-training or fine-tuning scene!')


if __name__ == '__main__':
    print('Our work Begin!')
    main()
    print('Our work Down!')


