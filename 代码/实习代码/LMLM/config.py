#!/user/bin/python3


def boolify(s):
    if s == 'True':
        return True
    if s == 'False':
        return False
    raise ValueError("huh?")


def autoconvert(s):
    for fn in (boolify, int, float):
        try:
            return fn(s)
        except ValueError:
            pass
    return s


def get_config(file_):
    config = {}
    with open(file_, "r") as f:
        data = f.readlines()
    for line in data:
        line = line.strip().split('=')
        if len(line) > 1:
            config[line[0]] = autoconvert(line[-1])
    return config


class Config(object):
    def __init__(self, conf_file):
        self.args = get_config(conf_file)
        if 'biderectional' not in self.args:
            self.args['biderectional'] = False
        if 'rnn_size' not in self.args:
            self.args['rnn_size'] = 200
        if 'num_layers' not in self.args:
            self.args['num_layers'] = 2
        if 'model_type' not in self.args:
            self.args['model_type'] = 'lstm'
        if 'is_train' not in self.args:
            self.args['is_train'] = True
        if 'batch_size' not in self.args:
            self.args['batch_size'] = 32
        if 'num_steps' not in self.args:
            self.args['num_steps'] = 20
        if 'max_epochs' not in self.args:
            self.args['max_epochs'] = 20
        if 'vocab_file' not in self.args:
            self.args['vocab_file'] = 'data/vocab.txt'
        self.args['vocab_size'] = 10000  # maybe updated according to vocab_file
        if 'save_dir' not in self.args:
            self.args['save_dir'] = 'model'
        if 'model_name' not in self.args:
            self.args['model_name'] = 'model.ckpt'
        if 'restore_epoch' not in self.args:
            self.args['restore_epoch'] = -1
        if 'test_file' not in self.args:
            self.args['test_file'] = 'data/test.txt'
        if 'nbest_file' not in self.args:
            self.args['nbest_file'] = 'data/nbest.txt'
        if 'score_file' not in self.args:
            self.args['score_file'] = 'data/score.txt'
        if 'dev_file' not in self.args:
            self.args['dev_file'] = 'data/dev.txt'
        if 'validation_interval' not in self.args:
            self.args['validation_interval'] = 1
        if 'init_scale' not in self.args:
            self.args['init_scale'] = 0.1
        if 'grad_clip' not in self.args:
            self.args['grad_clip'] = 5.0
        if 'learning_rate' not in self.args:
            self.args['learning_rate'] = 1.0
        if 'decay_rate' not in self.args:
            self.args['decay_rate'] = 0.5
        if 'keep_prob' not in self.args:
            self.args['keep_prob'] = 0.5
        if 'optimization' not in self.args:
            self.args['optimization'] = 'sgd'
