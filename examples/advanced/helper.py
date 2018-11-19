# Copyright (c) 2018, The SenseAct Authors.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import builtins
import tempfile, zipfile


def create_callback(shared_returns, load_model_data=None):
    builtins.shared_returns = shared_returns
    builtins.load_model_data = load_model_data

    def kindred_callback(locals, globals):
        shared_returns = globals['__builtins__']['shared_returns']
        if locals['iters_so_far'] == 0:
            if globals['__builtins__']['load_model_data'] is not None:
                tf_load_session_from_pickled_model(globals['__builtins__']['load_model_data'])
        else:
            ep_rets = locals['seg']['ep_rets']
            ep_lens = locals['seg']['ep_lens']
            if len(ep_rets):
                if not shared_returns is None:
                    shared_returns['write_lock'] = True
                    shared_returns['episodic_returns'] += ep_rets
                    shared_returns['episodic_lengths'] += ep_lens
                    shared_returns['write_lock'] = False
    return kindred_callback


def tf_load_session_from_pickled_model(load_model_data):
    """
    Restores tensorflow session from a zip file.
    :param load_model_path: A zip file containing tensorflow .ckpt and additional files.
    :return: None. Just restores the tensorflow session
    """
    import tensorflow as tf
    with tempfile.TemporaryDirectory() as td:
        arc_path = os.path.join(td, "packed.zip")
        with open(arc_path, "wb") as f:
            f.write(load_model_data['model'])

        zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)

        saver = tf.train.Saver()
        saver.restore(tf.get_default_session(), os.path.join(td, "model"))
