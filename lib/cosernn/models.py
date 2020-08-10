import tensorflow as tf

from cosernn.helpers import (
    day_count, hour_count, minute_count, device_count, top_context_count)


class CoSeRNN():
    def __init__(self, sample, rnnsize, batch_size):
        self.rnnsize = rnnsize
        self.batch_size = batch_size
        self.sample = sample

    def _make_rnn_gpu(self, input, num_layers, num_units, name, namespace="default", maxlen=None, init_state=None):
        '''
        RNN implementation using a CudnnLSTM
        :param input: a time series sinput
        :param num_layers: number of layers
        :param num_units:  number of hidden units
        :param name: component name
        :param namespace: namespace name
        :param maxlen: maximum length of the input
        :param init_state: specific initital state
        :return: Output of the LSTM
        '''
        input = tf.transpose(input, [1, 0, 2])
        with tf.variable_scope(namespace, reuse=tf.AUTO_REUSE) as scope:
            rnn = tf.contrib.cudnn_rnn.CudnnLSTM(
                num_layers=num_layers,
                num_units=num_units,
                dtype=tf.float32,
                name=name + "_rnn")

            if init_state is None:
                output, _ = rnn(input)
                output = tf.transpose(output, [1, 0, 2])
                return output

            i = []
            o = []
            i2 = []
            o2 = []
            for j in range(num_layers):
                internal = tf.layers.dense(init_state, num_units, name=name + "_internal_hist" + str(j))
                out = tf.layers.dense(init_state, num_units, name=name + "_out_hist" + str(j))
                internal = tf.expand_dims(internal, axis=0)
                out = tf.expand_dims(out, axis=0)

                i.append(internal)
                o.append(out)

                internal2 = tf.layers.dense(init_state, num_units, name=name + "_internal2_hist" + str(j))
                out2 = tf.layers.dense(init_state, num_units, name=name + "_out2_hist" + str(j))
                internal2 = tf.expand_dims(internal2, axis=0)
                out2 = tf.expand_dims(out2, axis=0)

                i2.append(internal2)
                o2.append(out2)

            internal = tf.concat(i, axis=0)
            out = tf.concat(o, axis=0)

            internal2 = tf.concat(i2, axis=0)
            out2 = tf.concat(o2, axis=0)

            internal_final = internal
            out_final = out

            initial_state = (internal_final, out_final)

            output, _ = rnn(input, initial_state=initial_state)
            output = tf.transpose(output, [1, 0, 2])

            return output

    def _extract_vals(self, item_emb, user_emb, user_f_emb):
        '''
        Extract a list of feature tensors from the generator.
        :param item_emb: embedding of all tracks
        :param user_emb: embedding of all users
        :param user_f_emb: embedding of user features for all users
        :return: list of feature tensors
        '''
        day_onehot, hour_onehot, minute_onehot, device_onehot, hist_avgs, mask, \
        mask_split, user, hist_skipped, hist_listened, histavgs_feedback, mask_split_above, time_since_last_session, top_context, number_of_tracks_in_sessions = self.sample[0], self.sample[1],self.sample[2], self.sample[3],self.sample[4],\
                                      self.sample[5],self.sample[6],self.sample[7],self.sample[8],self.sample[9],self.sample[10],self.sample[11], self.sample[12], self.sample[13], self.sample[14]

        mask_split_above = tf.cast(mask_split_above, tf.float32)

        day_onehot = tf.one_hot(day_onehot, day_count)
        hour_onehot = tf.one_hot(hour_onehot, hour_count)
        minute_onehot = tf.one_hot(minute_onehot, minute_count)
        device_onehot = tf.one_hot(device_onehot, device_count)

        EPS = 0.00001
        avoid_div_by_zero = tf.cast(tf.expand_dims(mask_split_above,-1) < 0.5, tf.float32) * EPS #
        avoid_div_by_zero_set_to_zero = tf.cast(tf.expand_dims(mask_split_above,-1) > 0.5, tf.float32) #

        hist_divider = tf.reduce_sum(tf.cast(hist_avgs>0, tf.float32), -1)
        hist_avgs = tf.nn.embedding_lookup(item_emb, hist_avgs)

        session_item_embs = hist_avgs
        session_item_feedback = histavgs_feedback

        hist_avgs = tf.reduce_sum(hist_avgs, axis=-2) / tf.expand_dims(hist_divider, -1)
        hist_avgs = hist_avgs / tf.expand_dims(tf.norm(hist_avgs + avoid_div_by_zero, ord=2, axis=-1), -1) * avoid_div_by_zero_set_to_zero

        hist_context = tf.concat([day_onehot, hour_onehot, device_onehot], axis=2) #

        hist_skipped_mask = tf.cast(hist_skipped > 0, tf.float32)
        hist_listened_mask = tf.cast(hist_listened > 0, tf.float32)

        hist_skipped = tf.nn.embedding_lookup(item_emb, hist_skipped)
        hist_listened = tf.nn.embedding_lookup(item_emb, hist_listened)

        hist_skip_divider = tf.reduce_sum(hist_skipped_mask, -1)
        skip_is0 = tf.cast(hist_skip_divider < 0.5, tf.float32)
        skip_is1 = tf.cast(hist_skip_divider > 0.5, tf.float32)
        hist_skipped = tf.reduce_sum(hist_skipped, axis=-2) / tf.expand_dims(hist_skip_divider + skip_is0*EPS, -1) # we add a small number since we can have sessions with no skips
        hist_skipped = hist_skipped / tf.expand_dims(tf.norm(hist_skipped + tf.expand_dims(skip_is0*EPS, -1), ord=2, axis=-1), -1) * tf.expand_dims(skip_is1, -1)

        hist_listened_divider = tf.reduce_sum(hist_listened_mask, -1)
        listen_is0 = tf.cast(hist_listened_divider < 0.5, tf.float32)
        listen_is1 = tf.cast(hist_listened_divider > 0.5, tf.float32)
        hist_listened = tf.reduce_sum(hist_listened, axis=-2) / tf.expand_dims(hist_listened_divider + listen_is0*EPS, -1)
        hist_listened = hist_listened / tf.expand_dims(tf.norm(hist_listened + tf.expand_dims(listen_is0*EPS, -1), ord=2, axis=-1), -1) * tf.expand_dims(listen_is1, -1)

        inputs = hist_avgs[:, :-1]
        inputs_contexts = hist_context[:, 1:]
        inputs_mask_loss = mask_split_above[:, 1:]
        inputs_mask = mask[:, 1:]
        targets = hist_avgs[:, 1:]

        inputs_skip = hist_skipped[:, :-1]
        inputs_listen = hist_listened[:, :-1]

        targets_skip = hist_skipped[:, 1:]
        targets_listen = hist_listened[:, 1:]

        session_item_embs = session_item_embs[:, 1:]
        session_item_feedback = session_item_feedback[:, 1:]

        time_since_last_session = time_since_last_session[:, 1:] # time since last session when predicting the current
        top_context_future = top_context[:, 1:]
        top_context = top_context[:, :-1] # top context belong to the 'previous' session, similar to inputs
        top_context = tf.one_hot(top_context, top_context_count)
        top_context_future = tf.one_hot(top_context_future, top_context_count)

        time_since_last_session = tf.expand_dims(time_since_last_session, -1)

        number_of_tracks_in_sessions = number_of_tracks_in_sessions[:, :-1]
        number_of_tracks_in_sessions = tf.expand_dims(number_of_tracks_in_sessions, -1)

        user_emb_vals = tf.nn.embedding_lookup(user_emb, user)
        user_f_emb_vals = tf.nn.embedding_lookup(user_f_emb, user)

        return inputs, inputs_contexts, inputs_mask, inputs_mask_loss, targets, \
                inputs_skip, inputs_listen, targets_skip, targets_listen, session_item_embs, session_item_feedback, user_emb_vals, time_since_last_session, top_context, number_of_tracks_in_sessions, user_f_emb_vals, top_context_future

    def _make_loss(self, preds, target, mask):
        '''
        Implements the cosine loss from the paper (assuming length normalized preds and targets)
        :param preds: predictions (normalized to unit length)
        :param target: targets (normalized to unit length)
        :param mask: potential mask if certain predictions are to be ignored
        :return: the cosine loss
        '''
        loss = (1.0 - tf.reduce_sum(target * preds, -1)) * mask
        return loss

    def _make_2_layers(self, logits, units, name_add=''):
        '''
        Implementation of 2 fully connected layers
        :param logits: the input to the layers
        :param units: number of hidden units
        :param name_add: unique name for the layers
        :return: Output of the last layer
        '''
        output = tf.layers.dense(logits, units, name=name_add+"_l1", reuse=tf.AUTO_REUSE, activation='relu')
        output = tf.layers.dense(output, units, name=name_add+"_l2", reuse=tf.AUTO_REUSE, activation='relu')
        output = tf.nn.dropout(output, keep_prob=self.keep_prob)
        return output

    def _embed_context(self, context, name, size, variational, activation="linear"):
        '''
        Embedding of the context features (either deterministic or sampled)
        :param context: context features
        :param name: unique name
        :param size: number of hidden units
        :param variational: True/False for using sampling or not
        :param activation: activation function
        :return: embedding of the context features
        '''
        mu = tf.layers.dense(context, size, name="mu_"+name, reuse=tf.AUTO_REUSE)
        std = tf.layers.dense(context, size, name="std_"+name, reuse=tf.AUTO_REUSE, activation='sigmoid')
        eps_std = tf.cond(self.is_training, lambda:1.0, lambda: 0.0)
        eps = tf.random.normal(tf.shape(std), dtype=tf.float32, mean=0., stddev=eps_std, name='epsilon')
        z = mu + tf.exp(std / 2) * eps

        if not variational:
            z = tf.layers.dense(context, size, name="trans_init_"+name, activation=activation, reuse=tf.AUTO_REUSE)

        return z

    def _sample_target(self, input, name, variational, name_add=''):
        '''
        Transforms the input using either a fully connected layer or through sampling.
        :param input: input tensor
        :param name: unique name
        :param variational: True/False depending on if output should be deterministic or sampling based
        :param name_add: unique name
        :return: Output of the transformation of the input, which is used as the predicted target in CoSeRNN
        '''
        mu = tf.layers.dense(input, 40, name=name_add+"_mu_"+name, reuse=tf.AUTO_REUSE)
        std = tf.layers.dense(input, 40, name=name_add+"_std_"+name, reuse=tf.AUTO_REUSE, activation='sigmoid')
        eps_std = tf.cond(self.is_training, lambda:1.0, lambda: 0.0)
        eps = tf.random_normal(tf.shape(std), dtype=tf.float32, mean=0., stddev=eps_std, name= name_add+'_epsilon')
        z = mu + tf.exp(std / 2) * eps

        if not variational:
            z = tf.layers.dense(input, 40, name=name_add+"_target_output"+name, reuse=tf.AUTO_REUSE)

        return z

    def _make_embedding(self, vocab_size, embedding_size, name, trainable=False, inittype=tf.random.normal):
        '''
        Initialize a embedding matrix
        :param vocab_size: number of unique elements to embed
        :param embedding_size: size of embedding
        :param name: unique name
        :param trainable: True/False to make trainable or not
        :return: embedding matrix
        '''
        W = tf.Variable(inittype(shape=[vocab_size, embedding_size]),
                        trainable=trainable, name=name)
        embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size])
        embedding_init = W.assign(embedding_placeholder)

        return W, embedding_placeholder, embedding_init

    def smooth_cosine_similarity(self, session_emb, sess_all_representations):
        '''
        Computes a smooth cosine value for the memory part of the model (not used in the RecSys paper).
        :param session_emb: embedding of a session
        :param sess_all_representations: embedding matrix of all keys (i.e. session embeddings) in the memory module
        :return: the cosine similarity to the closest key in the memory module
        '''
        sess_all_representations = tf.tile(tf.expand_dims(sess_all_representations, axis=0), multiples=[tf.shape(session_emb)[0], 1,1])
        session_emb = tf.expand_dims(session_emb, axis=2)
        inner_product = tf.matmul(sess_all_representations, session_emb)
        k_norm = tf.sqrt(tf.reduce_sum(tf.square(session_emb), axis=1, keep_dims=True))
        M_norm = tf.sqrt(tf.reduce_sum(tf.square(sess_all_representations), axis=2, keep_dims=True))
        norm_product = M_norm * k_norm
        similarity = tf.squeeze(inner_product / (norm_product + 1e-8), axis=2)

        return similarity

    def make_network(self, item_emb, user_emb, user_f_emb, is_training, is_single_sample, update_index, args, mm_multiplier):
        '''
        Creates the trainable network model
        :param item_emb: embedding matrix of items
        :param user_emb: embedding matrix of users
        :param user_f_emb:  embedding matrix of user features
        :param is_training: True/False depending on if Train/Test
        :param is_single_sample: True/False if the batch has a single sample only
        :param update_index: variable for the current index to be updated in the memory module (not used in the RecSys paper)
        :param args: model configuration dictionary
        :return: relevant tensorflow variables for training and tensorboard visualization
        '''
        self.dropout = args['dropout']
        self.is_training = is_training
        self.keep_prob = tf.cond(self.is_training, lambda: 1 - self.dropout, lambda: 1.0)

        if args['add_train_noise']:
            self.add_train_noise = tf.cond(self.is_training, lambda: 1.0, lambda: 0.0)
        else:
            self.add_train_noise = tf.cond(self.is_training, lambda: 0.0, lambda: 0.0)

        inputs, inputs_contexts, mask, inputs_mask_loss, targets, \
        inputs_skip, inputs_listen, targets_skip, targets_listen, session_item_embs, \
                    session_item_feedback, user_vals, time_since_last_session, top_context, number_of_tracks_in_sessions, user_f_emb_vals, top_context_future = self._extract_vals(item_emb, user_emb, user_f_emb)

        inputs_org = inputs # save the original inputs
        targets = targets_listen # we focus on predicting listened tracks during model training.

        user_f_emb_vals = tf.layers.dense(user_f_emb_vals, args['rnnsize'], activation='relu', name='init_features', reuse=tf.AUTO_REUSE)

        def context_rnn_run(inputs):
            choices = [inputs, inputs_contexts, number_of_tracks_in_sessions, time_since_last_session, top_context, top_context_future]
            chosen = [choices[cc] for cc in args['inputs_chosen']] # default = '0,1,2,3,4'
            inputs_combined = tf.concat(chosen, axis=-1)
            inputs_combined = self._embed_context(inputs_combined, 'context_sampler', args['rnnsize'], args['context_var'])

            if args['gpu']:
                hist_avgs_rnn = self._make_rnn_gpu(inputs_combined, 1, args['rnnsize'], 'histavg_rnn',
                                                   maxlen=tf.reduce_sum(mask, 1) , init_state=user_f_emb_vals)
            else:
                exit(-1)

            return hist_avgs_rnn

        hist_rnn_listen = context_rnn_run(inputs_listen)
        hist_rnn_skip = context_rnn_run(inputs_skip)
        hist_rnn_both = context_rnn_run(inputs_org)

        rep_choices = [hist_rnn_both, hist_rnn_skip, hist_rnn_listen]
        rep_chosen = [rep_choices[cc] for cc in args['session_rep']] # default = '0,1'
        hist_avgs_rnn = tf.concat(rep_chosen, axis=-1)

        avoid_div_by_zero = tf.cast(tf.expand_dims(inputs_mask_loss, -1) < 0.5, tf.float32) * 0.0000001
        rnn_pred = self._make_2_layers(hist_avgs_rnn, args['layersize'], name_add='1')
        rnn_pred = self._sample_target(rnn_pred, 'target_sampler', args['target_var'], name_add='1')
        rnn_pred_org = rnn_pred

        debugval = rnn_pred_org
        # Memory Module code start - this is not used in the RecSys paper.
        # Here we have: 1) The Spotify embedding and the output from the RNN model.
        # Use hist_avgs_rnn, at the correct index (the only 1 on the inputs_mask_loss mask), as the key for similarity matching

        user_pred = user_vals / (tf.expand_dims(tf.norm(user_vals, ord=2, axis=1), -1))
        key_emb, key_emb_ph, key_emb_init = self._make_embedding(args['MEMSIZE'], hist_avgs_rnn.shape[-1], 'key_emb', trainable=False, inittype=tf.zeros)
        session_emb, session_emb_ph, session_emb_init = self._make_embedding(args['MEMSIZE'], 40, 'session_emb', trainable=False, inittype=tf.zeros)
        error_emb, error_emb_ph, error_emb_init = self._make_embedding(args['MEMSIZE'], 1, 'error_emb', trainable=False, inittype=tf.zeros)

        if args['eval_mem_switch'] > -0.01:
            current_index = tf.cast(tf.argmax(inputs_mask_loss, -1), tf.int32)

            indexinto = tf.range(args['batch_size'])
            current_index = tf.concat((tf.expand_dims(indexinto, -1), tf.expand_dims(current_index, -1)), -1)

            current_target = tf.gather_nd(targets, current_index, name='current-target')
            current_key = tf.gather_nd(hist_avgs_rnn, current_index, name='current-key')

            # find most similar key compared to current_target
            cos_similarity = self.smooth_cosine_similarity(current_key, key_emb)  # [batch, n_session]
            neigh_sim, neigh_num = tf.nn.top_k(cos_similarity, k=args['mem_top_k'])  # [batch_size, memory_size]
            print(cos_similarity, neigh_sim, neigh_num)
            session_neighborhood = tf.nn.embedding_lookup(session_emb, neigh_num)  # [batch_size, memory_size, memory_dim]
            key_neighbourhood = tf.nn.embedding_lookup(key_emb, neigh_num)
            error_neighbourhood = tf.squeeze(tf.nn.embedding_lookup(error_emb, neigh_num), -1)

            neigh_sim = tf.expand_dims(tf.nn.softmax(neigh_sim), axis=2)

            mm_pred = tf.reduce_sum(neigh_sim * session_neighborhood, axis=1)
            mm_attn_w_key = tf.reduce_sum(neigh_sim * key_neighbourhood, axis=1)

            # Memory Module code end
            mm_pred = mm_pred
            rnn_pred = tf.gather_nd(rnn_pred, current_index, name='rnn-pred')

            mm_pred = tf.cond(tf.reduce_sum(mm_pred) > 0.01, lambda : mm_pred / (tf.expand_dims(tf.norm(mm_pred, ord=2, axis=1), -1)), lambda : tf.zeros(mm_pred.shape))
            rnn_pred = rnn_pred / (tf.expand_dims(tf.norm(rnn_pred, ord=2, axis=1), -1))

            # combine mm_pred and rnn_pred, based on their "keys"
            combined_keys = tf.concat((error_neighbourhood, mm_attn_w_key, current_key, mm_attn_w_key - current_key, tf.expand_dims(tf.reduce_sum(mm_attn_w_key*current_key, -1), 1)), axis=-1) #), axis=-1)

            combined_single = tf.layers.dense(combined_keys, 3, name='MEMORYNETWORK_alpha_combine_rnn_mm')
            combined_multiplier_exp = tf.expand_dims([0, 0, (-100 + 100*mm_multiplier)], axis=0)
            combined_multiplier_not_exp = tf.expand_dims([1, 1, mm_multiplier], axis=0)
            combined_single = combined_single + combined_multiplier_exp
            combined_single = tf.nn.softmax(combined_single, axis=-1)

            combined_single_0 = tf.expand_dims(combined_single[:, 0], -1)
            combined_single_1 = tf.expand_dims(combined_single[:, 1], -1)
            combined_single_2 = tf.expand_dims(combined_single[:, 2], -1)

            combined_pred_total = combined_single_0*user_pred + combined_single_1*rnn_pred + combined_single_2*mm_pred
            combined_pred_total = combined_pred_total / (tf.expand_dims(tf.norm(combined_pred_total, ord=2, axis=1), -1))

            combined_pred_total = tf.expand_dims(combined_pred_total, 1)

        multi_alpha = hist_avgs_rnn
        multi_alpha = tf.layers.dense(multi_alpha, 2, name='multi_alpha')
        multi_alpha = tf.nn.softmax(multi_alpha, axis=-1)

        multi_alpha_0 = tf.expand_dims(multi_alpha[:, :, 0], -1)
        multi_alpha_1 = tf.expand_dims(multi_alpha[:, :, 1], -1)

        # [rnn_pred, user_pred, rnn + user, softmax rnn + user]
        if args['pred_type'] == 0:
            multi_pred = rnn_pred_org
        elif args['pred_type'] == 1:
            multi_pred = tf.expand_dims(user_pred, 1) + 0*rnn_pred_org
        elif args['pred_type'] == 2:
            multi_pred = tf.expand_dims(user_pred, 1) + rnn_pred_org
        elif args['pred_type'] == 3: # default
            multi_pred = multi_alpha_0 * tf.expand_dims(user_pred, 1) + multi_alpha_1 * rnn_pred_org
        multi_pred = multi_pred / (tf.expand_dims(tf.norm(multi_pred + avoid_div_by_zero, ord=2, axis=2), -1))

        if args['eval_mem_switch'] < -0.01:
            combined_pred_total = multi_pred
            combined_single = multi_alpha

        loss_multi = self._make_loss(multi_pred, targets, inputs_mask_loss)
        loss = self._make_loss(combined_pred_total, targets, inputs_mask_loss)

        print(loss, loss_multi)

        loss_multi = tf.reduce_sum(loss_multi) / tf.reduce_sum(inputs_mask_loss)
        loss = tf.reduce_sum(loss) / tf.reduce_sum(inputs_mask_loss)

        org_avgcos = tf.reduce_sum(targets * combined_pred_total, -1) * inputs_mask_loss
        avgcos = tf.reduce_sum(org_avgcos) / tf.reduce_sum(inputs_mask_loss)

        org_avgcos_multi = tf.reduce_sum(targets * multi_pred, -1) * inputs_mask_loss
        avgcos_multi = tf.reduce_sum(org_avgcos_multi) / tf.reduce_sum(inputs_mask_loss)

        # compute dot product against items in session
        tmp = tf.expand_dims(combined_pred_total, axis=-2)
        session_item_scores = tf.reduce_sum(tmp * session_item_embs, axis=-1)

        # After reading, insert the current (key, target) into the module, where again target needs to be indexed by the correct index
        if args['eval_mem_switch'] > -0.01:
            key_emb_update = tf.scatter_update(key_emb, update_index, current_key)
            session_emb_update = tf.scatter_update(session_emb, update_index, current_target)
            current_index_single = tf.cast(tf.zeros(args['batch_size']), tf.int32)
            indexinto_single = tf.range(args['batch_size'])
            current_index_single = tf.concat((tf.expand_dims(indexinto_single, -1), tf.expand_dims(current_index_single, -1)), -1)
            current_pred = tf.gather_nd(combined_pred_total, current_index_single, name='current_pred')
            current_cosine = tf.expand_dims(tf.reduce_sum(current_pred * current_target, -1), -1)
            error_emb_update = tf.scatter_update(error_emb, update_index, current_cosine)
        else:
            key_emb_update = 0
            session_emb_update = 0
            error_emb_update = 0
            combined_single_0 = multi_alpha_0
            combined_single_1 = multi_alpha_0
            combined_single_2 = multi_alpha_0
            error_neighbourhood = 0

        # tensorboard vars
        sum1 = tf.summary.scalar("train cosine sim", avgcos)  # tf.reduce_sum(avgcos))
        sum1a = tf.summary.scalar("train cosine rnn+user", avgcos_multi)  # tf.reduce_sum(avgcos))

        sum2 = tf.summary.scalar("train loss", tf.reduce_mean(loss))
        sum2a = tf.summary.scalar("train loss rnn+user", tf.reduce_mean(loss_multi))

        multi_alpha_sum = tf.summary.scalar('multi_alpha--0', tf.reduce_mean(multi_alpha_0))

        combined_single_sum_0 = tf.summary.scalar('combined_single_sum--0', tf.reduce_mean(combined_single_0))
        combined_single_sum_1 = tf.summary.scalar('combined_single_sum--1', tf.reduce_mean(combined_single_1))
        combined_single_sum_2 = tf.summary.scalar('combined_single_sum--2', tf.reduce_mean(combined_single_2))

        summary_op = tf.summary.merge([sum1, sum2, sum1a, sum2a, multi_alpha_sum, combined_single_sum_0, combined_single_sum_1, combined_single_sum_2])

        pVar_val_loss = tf.placeholder(tf.float32, [])
        tmp = tf.Variable(0, dtype=tf.float32, name="tensorbardVar")
        update_pVar_val_loss = tmp.assign(pVar_val_loss)
        sum3 = tf.summary.scalar("val loss", tmp)

        pVar_val_avgcos = tf.placeholder(tf.float32, [])
        tmp = tf.Variable(0, dtype=tf.float32, name="tensorbardVar")
        update_pVar_val_avgcos = tmp.assign(pVar_val_avgcos)
        sum4 = tf.summary.scalar("val cosine sim", tmp)


        pVar_test_loss = tf.placeholder(tf.float32, [])
        tmp = tf.Variable(0, dtype=tf.float32, name="tensorbardVar")
        update_pVar_test_loss = tmp.assign(pVar_test_loss)
        sum5 = tf.summary.scalar("test loss", tmp)

        pVar_test_avgcos = tf.placeholder(tf.float32, [])
        tmp = tf.Variable(0, dtype=tf.float32, name="tensorbardVar")
        update_pVar_test_avgcos = tmp.assign(pVar_test_avgcos)
        sum6 = tf.summary.scalar("test cosine sim", tmp)

        # stuff for val mr and mrr
        pVar_val_mrr = tf.placeholder(tf.float32, [])
        tmp = tf.Variable(0, dtype=tf.float32, name="tensorbardVar")
        update_pVar_val_mrr = tmp.assign(pVar_val_mrr)
        sum7 = tf.summary.scalar("val mrr", tmp)

        pVar_val_mr = tf.placeholder(tf.float32, [])
        tmp = tf.Variable(0, dtype=tf.float32, name="tensorbardVar")
        update_pVar_val_mr = tmp.assign(pVar_val_mr)
        sum8 = tf.summary.scalar("val mr", tmp)

        # stuff for val mr and mrr
        pVar_test_mrr = tf.placeholder(tf.float32, [])
        tmp = tf.Variable(0, dtype=tf.float32, name="tensorbardVar")
        update_pVar_test_mrr = tmp.assign(pVar_test_mrr)
        sum9 = tf.summary.scalar("test mrr", tmp)

        pVar_test_mr = tf.placeholder(tf.float32, [])
        tmp = tf.Variable(0, dtype=tf.float32, name="tensorbardVar")
        update_pVar_test_mr = tmp.assign(pVar_test_mr)
        sum10 = tf.summary.scalar("test mr", tmp)

        summary_op_val = tf.summary.merge([sum3, sum4, sum7, sum8])
        summary_op_test = tf.summary.merge([sum5, sum6, sum9, sum10])

        return (
            debugval,
            loss,
            loss_multi,
            combined_pred_total,
            multi_pred,
            org_avgcos,
            org_avgcos_multi,
            loss,
            avgcos,
            loss,
            summary_op,
            summary_op_val,
            summary_op_test,
            [
                pVar_val_loss,
                pVar_val_avgcos,
                pVar_val_mrr,
                pVar_val_mr
            ],
            [
                update_pVar_val_loss,
                update_pVar_val_avgcos,
                update_pVar_val_mrr,
                update_pVar_val_mr
            ],
            [
                pVar_test_loss,
                pVar_test_avgcos,
                pVar_test_mrr,
                pVar_test_mr
            ],
            [
                update_pVar_test_loss,
                update_pVar_test_avgcos,
                update_pVar_test_mrr,
                update_pVar_test_mr
            ],
            avgcos,
            session_item_scores,
            targets,
            [
                key_emb_update,
                session_emb_update,
                error_emb_update
            ],
            [
                key_emb,
                key_emb_ph,
                key_emb_init
            ],
            [
                session_emb,
                session_emb_ph,
                session_emb_init
            ],
            [
                error_emb,
                error_emb_ph,
                error_emb_init
            ],
            combined_single_0,
            combined_single_1,
            error_neighbourhood,
            combined_single,
        )
