# Copyright 2020 Spotify AB
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import glob
import numpy as np
import os
import tensorflow as tf
import time

from cosernn.models import CoSeRNN
from cosernn.helpers import make_dataset_generator_onerow as make_dataset_generator


LASTTRACKS = [10, 25, 50, 100]
LASTSESSIONS = [10-1, 20-1, 30-1, 50-1]
RANK_L2 = False


def as_matrix(config):
    return [[k, str(w)] for k, w in config.items()]


def main():
    global item_vects, id2item, user_vects, id2user, random_tracks, random_sessions
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_records")
    parser.add_argument("--rnnsize", default=400, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--lr", default=0.0005, type=float)
    parser.add_argument("--decay_rate", default=0.99999, type=float)

    parser.add_argument("--context_var", default=0, type=int)
    parser.add_argument("--target_var", default=0, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--layersize', default=200, type=int)
    parser.add_argument('--add_train_noise', default=0, type=int)

    parser.add_argument('--imul', default=0.05, type=float)
    parser.add_argument('--MEMSIZE', default=2000, type=int)
    parser.add_argument('--mem_top_k', default=2, type=int)
    parser.add_argument('--eval_mem_switch', default=-1, type=int)

    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--POOLSIZE', default=10, type=int)

    parser.add_argument('--inputs_chosen', default='0,1,2,3,4', type=str)
    parser.add_argument('--session_rep', default='0,1', type=str)
    parser.add_argument('--pred_type', default=3, type=int)

    parser.add_argument('--inputs_chosen_options', default='[inputs, sampled_inputs_contexts, number_of_tracks_in_sessions, time_since_last_session, top_context, top_context_future]', type=str)
    parser.add_argument('--session_rep_options', default='[hist_rnn_both, hist_rnn_skip, hist_rnn_listen]', type=str)
    parser.add_argument('--pred_type_options', default='[rnn_pred, user_pred, rnn+user, softmax rnn+user]', type=str)

    parser.add_argument('--restorefile', default=None, type=str)
    parser.add_argument('--baselineEval', default=None, type=str)
    parser.add_argument('--max_length', default=400, type=int)
    args = parser.parse_args()

    args.context_var = args.context_var > 0.5
    args.target_var = args.target_var > 0.5
    args.add_train_noise = args.add_train_noise > 0.5
    args.gpu = args.gpu > 0.5

    args.inputs_chosen = [int(v) for v in args.inputs_chosen.split(',')]
    args.session_rep = [int(v) for v in args.session_rep.split(',')]
    args = vars(args)

    tfrecords = glob.glob(os.path.join(args["path_to_records"], "*"))
    num_items = 100
    num_users = 100

    tf.reset_default_graph()
    print('making session.....')
    with tf.Session() as sess:

        handle = tf.placeholder(tf.string, shape=[], name="handle_for_iterator")
        training_handle, train_iter, gen_iter = make_dataset_generator(sess, handle, args, tfrecords, 0)

        val_handle, val_iter, _ = make_dataset_generator(sess, handle, args, tfrecords, 1)
        test_handle, test_iter, _ = make_dataset_generator(sess, handle, args, tfrecords, 1)

        sample = gen_iter.get_next()
        mask = sample[11]

        model = CoSeRNN(sample, args['rnnsize'], args['batch_size'])

        item_emb, item_embedding_placeholder, item_embedding_init = model._make_embedding(num_items, 40, 'item_emb', trainable=False)
        user_emb, user_embedding_placeholder, user_embedding_init = model._make_embedding(num_users, 40, 'user_emb', trainable=False)
        user_f_emb, user_f_embedding_placeholder, user_f_embedding_init = model._make_embedding( num_users, 2, 'user_f_emb', trainable=False)

        mm_multiplier = tf.placeholder(tf.float32, name='mm_multiplier')
        is_training = tf.placeholder(tf.bool, name="is_training")
        is_single_sample = tf.placeholder(tf.bool,name="is_single_sample")  # set to True when the dataset with a one-hot mask instead of multi-hot
        update_index = tf.placeholder(tf.int32, shape=[None, ], name='update_index')


        debugval, loss_single, loss_multi, single_pred, multi_pred, \
        avgcos_single, avgcos_multi, targetmodel, histvalsmodel, context_now_org, summary_op, summary_op_val, summary_op_test, pVar_val, update_pVar_val, \
        pVar_test, update_pVar_test, avgcos_nvm, session_item_scores, avgtargets, \
        mem_updates, mem_key, mem_sess, mem_error, combined_alpha_rnn_mm, combined_alpha_spotify_other, error_neighbourhood, combined_single = model.make_network(
            item_emb, user_emb, user_f_emb, is_training, is_single_sample, update_index, args, mm_multiplier)


        step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(args["lr"],
                                        step,
                                        10000,
                                        args["decay_rate"],
                                        staircase=True, name="lr")
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, name="Adam")
        train_step = optimizer.minimize(loss_multi, global_step=step)

        init = tf.global_variables_initializer()
        sess.run(init)
        sess.run(train_iter.initializer)

        # In the experiments, we initialize with specific embeddings.
        #sess.run(item_embedding_init, feed_dict={item_embedding_placeholder: item_matrix})
        #sess.run(user_embedding_init, feed_dict={user_embedding_placeholder: user_matrix})
        #sess.run(user_f_embedding_init, feed_dict={user_f_embedding_placeholder: user_feature_matrix})

        loss = loss_multi
        avgcos = avgcos_multi

        eval_every = int(250 * args['imul'])
        patience_count = 0
        patience_max = 10
        iter_count = 0

        train_avgcos = []
        times = []
        losses_train = []

        runer4ever = True
        single_sample_val = False
        mm_multiplier_val = 0
        memory_index_batch = np.arange(args['batch_size']) % args['MEMSIZE']

        print('start training')
        while runer4ever:
            start = time.time()
            fd = {handle: training_handle, is_training: True, is_single_sample: single_sample_val, update_index: memory_index_batch.astype(np.float32), mm_multiplier: mm_multiplier_val}
            debugvalval, avgcosval,losses, _, maskval = sess.run([debugval, avgcos,loss, train_step, mask], feed_dict=fd)
            times.append(time.time() - start)

            train_avgcos += [np.mean(avgcosval[np.abs(avgcosval) > 0])]
            losses_train += [np.mean(losses)]
            iter_count += 1

            if iter_count % eval_every == 0:
                print("iteration", iter_count, 'avg time', np.mean(times))
                print('train, avg_cosine:', np.mean(train_avgcos), "loss:", np.mean(losses_train))

                train_avgcos = []
                times = []
                losses_train = []

                if patience_count > patience_max:
                    runer4ever = False


if __name__ == "__main__":
    main()
