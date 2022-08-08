import numpy as np
import tensorflow as tf
import numpy 
import sys
import pickle
from sst_config import Config
import time
from general_utils import Progbar
from keras.preprocessing.sequence import pad_sequences

vector_path='apparel_vectors'
path='parsed_data/apparel_dataset'
class Classifer(object):  
    def __init__(self, config, session):
        #inputs: features, mask, keep_prob, labels
        self.input_data = tf.placeholder(tf.int32, [None, None], name="inputs")
        self.labels=tf.placeholder(tf.int64, [None,], name="labels")
        self.mask=tf.placeholder(tf.int32, [None,], name="mask")
        self.dropout=self.keep_prob=keep_prob=tf.placeholder(tf.float32, name="keep_prob")
        self.rate=1-self.keep_prob
        self.config=config
        shape=tf.shape(self.input_data)
        #if sys.argv[4]=='lstm':
        #    self.dummy_input = tf.placeholder(tf.float32, [None, None], name="dummy")
        #embedding
        self.embedding=embedding = tf.Variable(tf.random_normal([config.vocab_size, config.hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="embedding", trainable=config.embedding_trainable)
        #apply embedding
        initial_hidden_states=tf.nn.embedding_lookup(embedding, self.input_data)
        #initial_cell_states=tf.identity(initial_hidden_states)

        initial_hidden_states = tf.nn.dropout(initial_hidden_states,self.rate)
        #initial_cell_states = tf.nn.dropout(initial_cell_states, self.rate)
        softmax_w = tf.Variable(tf.random_normal([config.hidden_size, config.num_label], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="softmax_w")
        softmax_b = tf.Variable(tf.random_normal([config.num_label], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="softmax_b")        
        print("Invalid model")
        representation = tf.reduce_mean(initial_hidden_states, axis = 1)   
        self.logits=logits = tf.matmul(representation, softmax_w) + softmax_b
        self.to_print=tf.nn.softmax(logits)
        #operators for prediction
        self.prediction=prediction=tf.argmax(logits,1)
        correct_prediction = tf.equal(prediction, self.labels)
        self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        
        #cross entropy loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=logits)
        self.cost=cost=tf.reduce_mean(loss)+ config.l2_beta*tf.nn.l2_loss(embedding)
        
        #designate training variables
        tvars=tf.trainable_variables()
        self.lr = tf.Variable(0.0, trainable=False)
        grads=tf.gradients(cost, tvars)
        grads, _ = tf.clip_by_global_norm(grads,config.max_grad_norm)
        self.grads=grads
        optimizer = tf.train.AdamOptimizer(config.learning_rate)        
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

def get_minibatches_idx(n, batch_size, shuffle=True):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    minibatch_start = 0
    for i in range(n // batch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + batch_size])
        minibatch_start += batch_size
    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])
    return minibatches

def run_epoch(session, config, model, data, eval_op, keep_prob, is_training):
    n_samples = len(data[0])
    print("Running %d samples:"%(n_samples))  
    minibatches = get_minibatches_idx(n_samples, config.batch_size, shuffle=False)

    correct = 0.
    total = 0
    total_cost=0
    prog = Progbar(target=len(minibatches))
    #dummynode_hidden_states_collector=np.array([[0]*config.hidden_size])

    to_print_total=np.array([[0]*2])
    for i, inds in enumerate(minibatches):
        x = data[0][inds]
        x=pad_sequences(x, maxlen=None, dtype='int32',padding='post', truncating='post', value=0.)
        y = data[1][inds]
        mask = data[2][inds]

        count, _, cost, to_print= \
        session.run([model.accuracy, eval_op,model.cost, model.to_print],\
            {model.input_data: x, model.labels: y, model.mask:mask, model.keep_prob:keep_prob})        	
        if not is_training:
            to_print_total=np.concatenate((to_print_total, to_print),axis=0)
        correct += count 
        total += len(inds)
        total_cost+=cost
        prog.update(i + 1, [("train loss", cost)])
    #if not is_training:
    #    print(to_print_total[:, 0].tolist())
    #    print(data[1].tolist())
    #    print(data[2].tolist())
    print("Total loss:")
    print(total_cost)
    accuracy = correct/total
    return accuracy

def train_test_model(config, i, session, model, train_dataset, valid_dataset, test_dataset):
    #compute lr_decay
    lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
    #update learning rate
    model.assign_lr(session, config.learning_rate * lr_decay)

    #training            
    print("Epoch: %d Learning rate: %.5f" % (i + 1, session.run(model.lr)))
    start_time = time.time()
    train_acc = run_epoch(session, config, model, train_dataset, model.train_op, config.keep_prob, True)
    print("Training Accuracy = %.4f, time = %.3f seconds\n"%(train_acc, time.time()-start_time))

    #valid 
    valid_acc = run_epoch(session, config, model, valid_dataset, tf.no_op(),1, True)
    print("Valid Accuracy = %.4f\n" % valid_acc)

    #testing
    start_time = time.time()
    test_acc = run_epoch(session, config, model, test_dataset, tf.no_op(),1, False)
    print("Test Accuracy = %.4f\n" % test_acc)    
    print("Time = %.3f seconds\n"%(time.time()-start_time))
    #return valid_acc, test_acc
def start_epoches(config, session,classifier, train_dataset, valid_dataset, test_dataset):
    #record max
    #max_val_acc=-1
    #max_test_acc=-1

    for i in range(config.max_max_epoch):
        train_test_model(config, i, session, classifier, train_dataset, valid_dataset, test_dataset)


def get_minibatches_idx(n, batch_size, shuffle=True):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // batch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + batch_size])
        minibatch_start += batch_size
    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])
    return minibatches

def prepare_data(seqs, labels):
    lengths = [len(s) for s in seqs]
    labels = numpy.array(labels).astype('int32')
    return [numpy.array(seqs), labels, numpy.array(lengths).astype('int32')]

def remove_unk(x, n_words):
    return [[1 if w >= n_words else w for w in sen] for sen in x] 

def load_data(path, n_words):
    with open(path, 'rb') as f:
        dataset_x, dataset_label= pickle.load(f)
        train_set_x, train_set_y = dataset_x[0], dataset_label[0]
        valid_set_x, valid_set_y =dataset_x[1], dataset_label[1]
        test_set_x, test_set_y = dataset_x[2], dataset_label[2]
    #remove unknown words
    train_set_x = remove_unk(train_set_x, n_words)
    valid_set_x = remove_unk(valid_set_x, n_words)
    test_set_x = remove_unk(test_set_x, n_words)

    return [train_set_x, train_set_y], [valid_set_x, valid_set_y], [test_set_x, test_set_y]

def word_to_vec(matrix, session,config, *args):
    
    print("word2vec shape: ", matrix.shape)
    for model in args:
        session.run(tf.assign(model.embedding, matrix))

if __name__ == "__main__":
    config = Config()
    #word2vec
    f = open(vector_path, 'rb')
    matrix= np.array(pickle.load(f))
    #print(matrix)
    config.vocab_size=matrix.shape[0]
    print(config.vocab_size) 

    #load datasets
    train_dataset, valid_dataset, test_dataset = load_data(\
        path=path,n_words=config.vocab_size)
    #print(train_dataset[0])
    config.num_label= len(set(train_dataset[1]))
    print("number label: "+str(config.num_label))    
    train_dataset = prepare_data(train_dataset[0], train_dataset[1])
    valid_dataset =prepare_data(valid_dataset[0],valid_dataset[1])
    test_dataset =prepare_data(test_dataset[0],test_dataset[1])
    with  tf.Session() as session:
        initializer=tf.random_normal_initializer(0,0.05)
        classifier=Classifer(config=config,session=session)
        total=0
        for v in tf.trainable_variables():            
            print(v.name)
            shape=v.get_shape()
            print(shape)           
            try:
                size=shape[0].value*shape[1].value 
            except:
                size=shape[0].value
                print(size)
            total+=size
        print(total)
        #input()
        init=tf.global_variables_initializer()
        session.run(init)
        word_to_vec(matrix,session,config,classifier)
        #print(classifier.embedding)
        start_epoches(config, session,classifier, train_dataset, valid_dataset, test_dataset)        












"""
    n_samples = len(train_dataset[0])
    minibatches = get_minibatches_idx(n_samples, 5, shuffle=False)
    #print(minibatches)
    prog = Progbar(target=len(minibatches))
    #print(prog)
    #print(list(enumerate(minibatches)))
    for i, inds in enumerate(minibatches):
        x = train_dataset[0][inds]
        #print(x)
        x=pad_sequences(x, maxlen=700, dtype='int32',padding='post', truncating='post', value=0.)        
        #print(x)
        embedding = tf.Variable(tf.random_normal([config.vocab_size, config.hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="embedding", trainable=config.embedding_trainable)
        initial_hidden_states=tf.nn.embedding_lookup(embedding, x)
        initial_cell_states=tf.identity(initial_hidden_states)
        initial_hidden_states = tf.nn.dropout(initial_hidden_states,0.5)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(sess.run(initial_hidden_states))
            print('print next:')
            #print(sess.run(initial_cell_states))
        #print(initial_hidden_states)
        input()
        
        print('print:y')
        y = train_dataset[1][inds]
        print(y)
        print('print:mask')
        mask = train_dataset[2][inds]
        print(mask)
        print('next')
"""



















"""
    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_normal_initializer(0, 0.05)

        total=0
        #print trainable variables
        print(tf.trainable_variables())
        input()
        for v in tf.trainable_variables():
            print(v.name)
            input()
            shape=v.get_shape()
            try:
                size=shape[0].value*shape[1].value
            except:
                size=shape[0].value
            total+=size
        print(total)
        #initialize
        init = tf.global_variables_initializer()

        session.run(init)
"""








"""
import os
import collections
import pickle
import gensim
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import re
import random
import numpy as np
import sys
import os 
def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        print('made directory %s' % path)
make_dir("parsed_data")

def text_preprocessing(datasets):
    dataset_text=[]
    dataset_label=[]
    for file in datasets:
        lines=[]
        labels=[]
        with open(file) as f:
            for l in f:
                try:
                    words=re.split('\s|-',l.lower().split("|||")[0].strip())
                    #words=words
                    label=int(l.lower().split("|||")[1].strip())
                    lines+=[words]
                    labels+=[label]
                    
                except:
                    continue
        dataset_text+=[lines]
        dataset_label+=[labels]
    return dataset_text, dataset_label

def insert_word(dataset,all_words):
    for lines in dataset:
        for l in lines:
            all_words+=l


def convert_word_to_number(dataset_text,dataset_label,common_word):
    transformed_text=[]
    transformed_lable=[]
    for lines,labels in zip(dataset_text,dataset_label):
        new_x=[]
        new_lable=[]
        for l,label in zip(lines,labels):
            for w in l:
                if w in common_word:
                    words=common_word[w]
                else:
                    words=1
                new_x+=[words]
                new_lable+=[label]
        transformed_text+=[new_x]
        transformed_lable+=[new_lable]
    return transformed_text,transformed_lable


if __name__ == "__main__":
    datasets=['sst_data/'+sys.argv[1]+'_trn', 'sst_data/'+sys.argv[1]+'_dev', 'sst_data/'+sys.argv[1]+'_tst']
    dataset_text, dataset_label=text_preprocessing(datasets)
    all_words=[]
    insert_word(dataset_text,all_words)
    counter=collections.Counter(all_words)
    #print(counter)
    vocab=len(counter)
    #print(vocab)
    vocab_size=vocab-2
    common_word=dict(counter.most_common(vocab_size))
    #print(common_word)
    print(len(common_word))
    c=2
    for key in common_word:
        common_word[key]=c
        c+=1
    #print(common_word) 
    transformed_text,transformed_lable=convert_word_to_number(dataset_text,dataset_label,common_word)
    #print(transformed_text)
    #print(transformed_lable)
    pickle.dump((transformed_text,transformed_lable),open('parsed_data/'+sys.argv[1]+'_dataset','wb'))
    glove_filename="./glove.6B/glove.6B.50d.txt"
    glove_dict={}
    with open(glove_filename) as f:
        for line in f:
            line=line.strip().split(' ')
            word=line[0]
            embedding=[]
            for x in line[1:]:
                embedding+=[float(x)]
            glove_dict[word]=embedding
            #print(glove_dict)
            #g=input()
    word2vec=[np.random.normal([1,50]).tolist(),np.random.normal([1,50]).tolist()]
    #print(word2vec)
    #g=input()
    missimg=0
    for number,word in sorted(zip(common_word.values(),common_word.keys())):
        try:
            word2vec.append(glove_dict[word])
        except KeyError:
            missimg+=1
            word2vec.append(np.random.normal([1,50]).tolist())
        #print('missimg=',missimg)
        #print('word2vec=',word2vec)
        #g=input()
        #print(len(word2vec))
    #print(word2vec)
    #print(len(word2vec))
    print('word2vec[0]=',word2vec[0])
    print ('word2vec[1]=',word2vec[1])
    print('word2vec[2]=',word2vec[2])
    print('word2vec[50]=',word2vec[50])
    print('word2vec[70]=',word2vec[70])
    print(missimg)
    pickle.dump(word2vec,open(sys.argv[1]+'_vectors','wb'))
    print(np.array(word2vec).shape)

"""
   
    