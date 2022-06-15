# just a simple test for load and  save to checkpoint to jit script

import argparse
import os

import yaml
from models.rnnlm import init_lm_model
from utils.checkpoint import save_checkpoint


def get_args():
    parser = argparse.ArgumentParser(description='export your script model')
    parser.add_argument('--config', required=True, help='config file')
    # parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    # parser.add_argument('--output_file', default=None, help='output file')
    # parser.add_argument('--output_quant_file',
    #                     default=None,
    #                     help='output quantized model file')
    args = parser.parse_args()
    return args


args = get_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
with open(args.config, 'r') as fin:
    configs = yaml.load(fin, Loader=yaml.FullLoader)

configs['vocab_size'] = 100
model = init_lm_model(configs)
print(model)
# RNNLM(
#   (model): RNNEncoder(
#     (lookup_table): Embedding(100, 100)
#     (stacked_rnn): StackedRNNLayer(
#       (dropout): Dropout(p=0.0, inplace=False)
#     )
#     (log_softmax): AdaptiveLogSoftmax(
#       (head): Linear(in_features=20, out_features=12, bias=True)
#       (tail): ModuleList(
#         (0): Sequential(
#           (0): Linear(in_features=20, out_features=5, bias=False)
#           (1): Linear(in_features=5, out_features=20, bias=False)
#         )
#         (1): Sequential(
#           (0): Linear(in_features=20, out_features=5, bias=False)
#           (1): Linear(in_features=5, out_features=70, bias=False)
#         )
#       )
#     )
#   )
#   (criterion): LabelSmoothingLoss(
#     (criterion): KLDivLoss()
#   )
# )
save_checkpoint(model, "tools/test.pt")
# tools/test.pt
