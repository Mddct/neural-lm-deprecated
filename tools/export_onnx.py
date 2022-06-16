from __future__ import print_function

import argparse
import os
import sys

import torch
import yaml
from models.rnnlm import init_lm_model
from utils.checkpoint import load_checkpoint

try:
    import onnx
except ImportError:
    print('Please install onnx!')
    sys.exit(1)


def get_args():
    parser = argparse.ArgumentParser(description='export your script model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--output_path',
                        required=True,
                        help='output directory')
    args = parser.parse_args()
    return args


def export_encoder(asr_model, args):
    model = asr_model
    model.forward = asr_model.forward_step
    input = torch.tensor([[1]], dtype=torch.int64)  # [bs, step=1]
    output = torch.tensor([[2]], dtype=torch.int64)  # [bs, step=1]

    state_m, state_c = model.zero_states(1)
    inputs = (input, output, state_m, state_c)

    dynamic_axes = {
        'ilbale': {
            0: "B"
        },
        'olabel': {
            0: "B"
        },
        "state_m": {
            0: "B"
        },
        "state_c": {
            0: "B"
        },
        "score": {
            0: "B"
        },
        "new_state_m": {
            0: "B"
        },
        "new_state_c": {
            0: "B"
        }
    }
    torch.onnx.export(model,
                      inputs,
                      args.output_path,
                      opset_version=14,
                      export_params=True,
                      do_constant_folding=True,
                      input_names=['ilabel', 'olabel', 'state_m', 'state_c'],
                      output_names=['score', 'new_state_m', 'new_state_c'],
                      dynamic_axes=dynamic_axes,
                      verbose=True)
    onnx_encoder = onnx.load(args.output_path)
    _ = onnx_encoder


def main():
    torch.manual_seed(777)
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    # TODO: change later from train.yaml
    configs['vocab_size'] = 100
    model = init_lm_model(configs)
    load_checkpoint(model, args.checkpoint)
    model.eval()
    export_encoder(model, args)


if __name__ == '__main__':
    main()
