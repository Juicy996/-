import numpy as np
import tensorflow as tf
import argparse
import time
import os
import math
from data import TextLoader
from vocab import Vocab
from lm import LM
from config import Config
import sys
import shutil

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_file', type=str, default='rescore.conf',
                        help="configure file")
    args = parser.parse_args()
    config = Config(args.conf_file)
    rescore(config.args)

def run_epoch(session, model, data,data_nbest, data_loader, biderectional=False):
    log_probs = list()
    state = session.run(model.initial_lm_state)
    batch_iter = data_loader.nbest_data_iterator(data,
                                                 model.batch_size,
                                                 biderectional=biderectional)

    batch_iter_nbest = data_loader.nbest_data_iterator(data,
                                                 model.batch_size*4,
                                                 biderectional=biderectional)

    # for (inputs, inputs_reverse),(inputs_nbest,inputs_nbest_reverse) in zip(batch_iter,batch_iter_nbest):
    for value0,value1 in zip(batch_iter,batch_iter_nbest):
        inputs, inputs_reverse = value0
        input(len(inputs))
        # sent_num = model.batch_size * step
        # if sent_num % 1024 == 0:
        #     print("rescored %d sentences " % sent_num)
        seq_len = list(map(lambda x: len(x), inputs))
        max_len = max(seq_len)
        for i in range(len(seq_len)):
            cur_len = seq_len[i]
            while cur_len < max_len:
                inputs[i].append(int(0))
                if biderectional:
                    inputs_reverse[i].append(int(0))
                cur_len += 1

        
        probs, _, _ = session.run([model.softmax, model.final_state, tf.no_op()],
                                    {model.input_data: inputs,
                                    model.initial_lm_state: state})
        for b in range(model.batch_size):
            sent_log_p = 0.0
            cur_len = seq_len[b]
            for i in range(cur_len):
                if i == cur_len - 1:
                    if biderectional:
                        wid = inputs_reverse[b][i-1]
                    else:
                        wid = int(2)
                else:
                    wid = inputs[b][i+1]
                word_prob = probs[b * max_len + i][wid]
                if word_prob == 0:
                    word_prob = 1e-08
                sent_log_p += math.log(word_prob)

            log_probs.append(sent_log_p)
        # input(len(log_probs))
    return log_probs

def hinge_loss(log_probs,log_probs_nbest):
    result = log_probs_nbest.reshape((len(log_probs), 4))
    loss = 0
    for i in range(0,len(log_probs)):
        for j in range(0,4):
            loss+= max(0,log_probs[i] - result[i][j])
    return loss



def rescore(args):
    start = time.time()
    model_name = args['model_name']
    restore_epoch = args['restore_epoch']
    save_dir = args['save_dir']
    shutil.copy(args['conf_file'], save_dir)

    vocab = Vocab(args['vocab_file'])
    args['vocab_size'] = vocab.size
    print("Word vocab size: " + str(vocab.size))

    data_loader = TextLoader(args, train=False)
    data_nbest = data_loader.read_nbestdata(args['nbest_file'])
 
    data = data_loader.read_data(args['train_file'])

    score_file = args['score_file']
    print("Begin rescoring...")

    with tf.Graph().as_default(), tf.Session() as sess:
        with tf.variable_scope("lm", reuse=None):
            model = LM(args, is_training=False, is_rescoring=True)
        saver = tf.train.Saver(tf.all_variables())
        tf.global_variables_initializer().run()

        # log_probs = run_epoch(sess, model, data, data_loader, args['biderectional'])
        log_probs_nbest = run_epoch(sess, model,data, data_nbest, data_loader, args['biderectional'])

        loss = hinge_loss(log_probs,log_probs_nbest)
        

        print(len(log_probs))
        print(len(log_probs_nbest))

    

if __name__ == '__main__':
    main()