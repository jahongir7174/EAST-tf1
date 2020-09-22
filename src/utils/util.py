import re
import sys
import tensorflow as tf

class ProgressBar(object):
    STYLE = 'Epoch %(epoch)2d %(bar)s  m_loss %(m_loss)f t_loss %(t_loss)f'

    def __init__(self, total, width=40, style=STYLE, symbol='=', output=sys.stderr):
        assert len(symbol) == 1
        self.current = 0
        self.total = total
        self.width = width
        self.symbol = symbol
        self.output = output
        self.style = re.sub(r"(?P<name>%\(.+?\))d", r'\g<name>%dd' % len(str(total)), style)

    def __call__(self, epoch, m_loss, t_loss):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'
        args = {'bar': bar, 'm_loss': m_loss, 't_loss': t_loss, 'epoch': epoch}
        print('\r' + self.style % args, file=self.output, end='')

    def done(self, epoch, m_loss, t_loss):
        self.current = self.total
        self(epoch=epoch, m_loss=m_loss, t_loss=t_loss)
        print('', file=self.output)


def get_nb_weights():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters
