#!/usr/bin/python3

import numpy as np
import tensorflow as tf
import argparse
import time
import os
import codecs
from data import TextLoader
from vocab import Vocab
from lm import LM
from config import Config
import shutil


os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_file', type=str, default='train.conf',
                        help="configure file")
    train_args = parser.parse_args()
    lm_config = Config(train_args.conf_file)
    train(lm_config.args)

def run_epoch(session, models, data_loader, args,
              init_state_tensors, init_state_values, eval_op, train_perplexity, final_state_tensors,
              apply_gradient_op, lr, lr_value, e):
    perplexity = 0.0
    batch_size = args['batch_size']
    num_steps = args['num_steps']
    # if eval_op == "test":
    #     num_steps = 1
    #     batch_size = 1
    gpu_num = args['gpu_num']
    step_ = 0
    if eval_op == "train":
        data = data_loader.read_dataset(args['train_file'], 0)
        
        data_nbest = data_loader.read_data(args['nbest_file'])
        
    else:
        data = data_loader.dev_data
        data_nbest = data_loader.read_data(args['nbest_file'])
    data_gen = data_loader.data_iterator(data,
                                         batch_size * gpu_num,
                                         num_steps,
                                         biderectional=args['biderectional'])

    # data_gen_nbest = data_loader.nbest_data_iterator(data_nbest,
    #                                      batch_size * gpu_num*4,
    #                                      biderectional=args['biderectional'])

    data_gen_nbest = data_loader.data_iterator(data_nbest,
                                         batch_size * gpu_num*4,
                                         num_steps,
                                         biderectional=args['biderectional'])

    try:
        # for step, batch in enumerate(data_gen, start=1):
        for batch, batch_nbest in zip(data_gen,data_gen_nbest):
            
            feed_dict = {t: v for t, v in zip(init_state_tensors, init_state_values)}
            
            for k in range(gpu_num):
                model = models[k]
                start = k * batch_size
                end = (k + 1) * batch_size
                start_nbest = k * batch_size*4
                end_nbest = (k + 1) * batch_size*4
                feed_dict.update(get_feed_dict_from_data(batch,
                                                         batch_nbest,
                                                         start,
                                                         end,
                                                         start_nbest,
                                                         end_nbest,
                                                         model,
                                                         biderectional=args['biderectional']))
            feed_dict.update({lr: lr_value})
        
            if eval_op == "train":
                ret = session.run([train_perplexity, final_state_tensors,
                                   apply_gradient_op, lr],
                                  feed_dict=feed_dict)
            else:
                ret = session.run([train_perplexity, final_state_tensors], feed_dict=feed_dict)
                init_state_values = ret[1]
            
        
            perplexity += ret[0]
            # input(perplexity)
            # inputs, _, _ = batch
            # step_ += sum(list(map(lambda x: len(x), inputs)))
    except StopIteration:
        print("Finished!")
    # return np.exp(perplexity / step_)
    return perplexity


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def get_feed_dict_from_data(sample, sample_nbest,start, end,start_nbest,end_nbest, model, biderectional=False):
    feed_dict = dict()
    x, y, z = sample

    x_nbest, y_nbest, z_nbest = sample_nbest
    x_nbest = x_nbest[start_nbest:end_nbest]
    y_nbest = y_nbest[start_nbest:end_nbest]
    x_nbest_1, x_nbest_2, x_nbest_3, x_nbest_4 = [], [], [], []
    y_nbest_1, y_nbest_2, y_nbest_3, y_nbest_4 = [], [], [], []
    for i in range(0,len(x_nbest),4):
        x_nbest_1.append(x_nbest[i])
        x_nbest_2.append(x_nbest[i+1])
        x_nbest_3.append(x_nbest[i+2])
        x_nbest_4.append(x_nbest[i+3])

    for i in range(0,len(y_nbest),4):
        y_nbest_1.append(y_nbest[i])
        y_nbest_2.append(y_nbest[i+1])
        y_nbest_3.append(y_nbest[i+2])
        y_nbest_4.append(y_nbest[i+3])

    # input(x_nbest_1)
    feed_dict[model.input_data] = x[start:end]
    feed_dict[model.targets] = y[start:end]
    feed_dict[model.input_data_nbest_1] = x_nbest_1
    feed_dict[model.input_data_nbest_2] = x_nbest_2
    feed_dict[model.input_data_nbest_3] = x_nbest_3
    feed_dict[model.input_data_nbest_4] = x_nbest_4

    feed_dict[model.targets_nbest_1] = y_nbest_1
    feed_dict[model.targets_nbest_2] = y_nbest_2
    feed_dict[model.targets_nbest_3] = y_nbest_3
    feed_dict[model.targets_nbest_4] = y_nbest_4
    # feed_dict[model.targets_nbest] = y[start_nbest:end_nbest]
    
   

    if biderectional:
        feed_dict[model.input_data_reverse] = z[start:end]
    return feed_dict

# def get_feed_dict_from_data(sample, start, end, model, biderectional=False):
#     feed_dict = dict()
#     x, y, z = sample
  
#     feed_dict[model.input_data] = x[start:end]
#     feed_dict[model.targets] = y[start:end]
 
#     if biderectional:
#         feed_dict[model.input_data_reverse] = z[start:end]
#     return feed_dict

def train(args):
    start = time.time()
    save_dir = args['save_dir']
    try:
        os.stat(save_dir)
    # except FileNotFoundError:
    except IOError:
        os.mkdir(save_dir)
    shutil.copy(args['conf_file'], save_dir)
    model_name = args['model_name']
    out_file = os.path.join(args['save_dir'], args['log_file'])
    f_out = codecs.open(out_file, "w", encoding="UTF-8")

    vocab = Vocab(args['vocab_file'])
    args['vocab_size'] = vocab.size
    f_out.write("Word vocab size: " + str(vocab.size) + "\n")
    print("vocab size: " + str(vocab.size) + "\n")

    data_loader = TextLoader(args)
    # data = data_loader.read_nbestdata(args['nbest_file'])
    
    # ist_shape = np.array(data).shape
    # print(ist_shape)

    f_out.write("Begin training...\n")
    print("Begin training...")

    with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        dev_pp = 10000000.0
        best_epoch = 0
        e = 0
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        max_epochs = args['max_epochs']
        f_out.write("max_epochs=" + str(max_epochs) + "\n")
        gpu_num = args['gpu_num']
        learning_rate = args['learning_rate']
        tower_grads = []
        models = []
        models_dev = []
        total_losses = tf.get_variable('total_losses', [],
                                       initializer=tf.constant_initializer(0.0),
                                       trainable=False)
        total_losses_dev = tf.get_variable('total_losses_dev', [],
                                           initializer=tf.constant_initializer(0.0),
                                           trainable=False)
        lr = tf.placeholder(tf.float32, shape=[])
        opt = tf.train.GradientDescentOptimizer(lr)

        for i in range(gpu_num):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % ('tower', i)):
                    with tf.variable_scope('lm', reuse=i > 0):
                        lm_model = LM(args, is_training=True)
                        # loss = lm_model.cost
                        loss = lm_model.loss
                        models.append(lm_model)
                        total_losses += loss
                        tvars = tf.trainable_variables()
                        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), args['grad_clip'])
                        tower_grads.append(zip(grads, tvars))

        for i in range(gpu_num):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % ('tower', i)):
                    with tf.variable_scope('lm', reuse=True):
                        lm_model_dev = LM(args, is_training=False)
                        loss_dev = lm_model_dev.loss
                        models_dev.append(lm_model_dev)
                        total_losses_dev += loss_dev
        with tf.device("/cpu:0"):
            grad_avr = average_gradients(tower_grads)
            apply_gradient_op = opt.apply_gradients(grad_avr, global_step=global_step)  
            train_perplexity = total_losses
            dev_perplexity = total_losses_dev

        tf.global_variables_initializer().run()

        init_state_tensors = []
        final_state_tensors = []
        for model in models:
            init_state_tensors.extend(model.initial_lm_state)
            final_state_tensors.extend(model.final_state)
        init_state_values = sess.run(init_state_tensors)
        init_state_tensors_dev = []
        final_state_tensors_dev = []
        for model in models_dev:
            init_state_tensors_dev.extend(model.initial_lm_state)
            final_state_tensors_dev.extend(model.final_state)
        init_state_values_dev = sess.run(init_state_tensors_dev)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        restore_epoch = args['restore_epoch']
        if restore_epoch >= 0:
            model_path = os.path.join(save_dir, model_name + "-" + str(restore_epoch))
            f_out.write("restore model: " + str(model_path) + "\n")
            saver.restore(sess, model_path)
            e = restore_epoch + 1
        while e < max_epochs:
            train_pp_ = run_epoch(sess, models, data_loader, args,
                                  init_state_tensors, init_state_values, "train", train_perplexity, final_state_tensors,
                                  apply_gradient_op, lr, learning_rate, e)
            # input(sess.run(models.logits))
           
            dev_pp_ = run_epoch(sess, models_dev, data_loader, args,
                                init_state_tensors_dev, init_state_values_dev, "test", dev_perplexity, final_state_tensors_dev,
                                apply_gradient_op, lr, learning_rate, e)

            f_out.write("Epoch: %d\n" % (e + 1))
            f_out.write("train ppl: %.3f\n" % train_pp_)
            f_out.write("dev ppl: %.3f\n" % dev_pp_)
            f_out.flush()

if __name__ == '__main__':
    main()
