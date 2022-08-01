import os
import json
import sys
try:
    import tensorboardX as tbx
except ImportError as e:
    print("TensorboardX not installed. No tensorboard logging.")
    tbx = None

class Logger(object):
    def __init__(self, filename="log.txt", mode='a'):
        self.terminal = sys.stdout
        self.log = open(filename, mode)
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        self.log.flush()

def add_tf_summary_value(writer, key, value, iteration=None):
    writer.add_scalar(tag=key, scalar_value=value, walltime=iteration)

def add_tf_summary_image(writer, key, image, iteration=None):
    # image: in CHW order
    writer.add_image(tag=key, img_tensor=image, walltime=iteration)

def add_tf_summary_histogram(writer, key, array, iteration=None):
    writer.add_histogram(tag=key, values=array, global_step=iteration)

def add_tf_summary_text(writer, key, text, iteration=None):
    writer.add_text(tag=key, text_string=text, global_step=iteration)

def save_options_cmdline(output_dir, args):
    # save options
    with open(os.path.join(output_dir, 'config.txt'), 'w') as outfile:
        outfile.write(json.dumps(vars(args), indent=2))

    # save cmdline
    with open(os.path.join(output_dir, 'cmdline.txt'), 'w') as outfile:
        outfile.write('python ' + ' '.join(sys.argv))

def set_logger(output_dir):
    # set logger
    logger = Logger(filename=os.path.join(output_dir, 'log.txt'), mode='w')
    sys.stdout = logger
    return logger

def set_tb_logger(output_dir):
    # set tb_logger
    tb_logger = tbx.SummaryWriter(output_dir) if tbx else None
    return tb_logger

def tb_save_options_cmdline(tb_logger, args):
    # save options and cmdline in tf_logging
    if tb_logger:
        add_tf_summary_text(tb_logger, "config", json.dumps(vars(args), indent=2))
        add_tf_summary_text(tb_logger, "cmdline", 'python ' + ' '.join(sys.argv))