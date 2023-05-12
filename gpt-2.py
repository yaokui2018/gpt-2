# -*- coding: utf-8 -*-
# Author: 薄荷你玩
# Date: 2023/05/11

import os
from keras_gpt_2 import load_trained_model_from_checkpoint, get_bpe_from_files, generate


model_folder = 'models/124M'
config_path = os.path.join(model_folder, 'hparams.json')
checkpoint_path = os.path.join(model_folder, 'model.ckpt')
encoder_path = os.path.join(model_folder, 'encoder.json')
vocab_path = os.path.join(model_folder, 'vocab.bpe')


print('Load model from checkpoint...')
model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
print('Load BPE from files...')
bpe = get_bpe_from_files(encoder_path, vocab_path)
print('Generate text...')

output = generate(model, bpe, ['hello,'], length=50, top_k=1)
print(output[0])

while True:

    output = generate(model, bpe, [input(">> ")], length=50, top_k=1)
    print(output[0])
