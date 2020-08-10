import numpy as np
import tensorflow as tf

PLATFORMS = {'mobile':0, 'desktop':1, 'speaker':2, 'web':3, 'tablet':4, 'tv':5, 'remaining':6}
SPOTIFY_TOP_CONTEXT = {
        'nan': 0,
        'play_queue': 1,
        'personalized_playlist': 2,
        'mix': 3,
        'catalog': 4,
        'editorial_playlist': 5,
        'radio': 6,
        'algotorial_playlist': 7,
        'user_playlist': 8,
        'sounds_of': 9,
        'charts': 10,
        'user_collection': 11
                       }
day_count = 7
hour_count = 24
minute_count = 4
device_count = len(PLATFORMS)
top_context_count = len(SPOTIFY_TOP_CONTEXT)

def extract_fn_onerow(data_record):
    features = {
        'day_hist': tf.FixedLenFeature([400], tf.int64),
        'hour_hist': tf.FixedLenFeature([400], tf.int64),
        'minute_hist': tf.FixedLenFeature([400], tf.int64),
        'device_hist': tf.FixedLenFeature([400], tf.int64),

        'hist_avgs': tf.VarLenFeature(tf.int64),
        'histavgs_feedback': tf.VarLenFeature(tf.int64),
        'histavgs_skip': tf.VarLenFeature(tf.int64),
        'histavgs_listen': tf.VarLenFeature(tf.int64),

        'hist_avgs_shape': tf.FixedLenFeature([2], tf.int64),

        'user': tf.FixedLenFeature([1], tf.int64),
        'mask': tf.FixedLenFeature([400], tf.int64),
        'mask_split': tf.FixedLenFeature([400], tf.int64),
        'mask_split_above': tf.FixedLenFeature([400], tf.int64),


        'time_since_last_session': tf.FixedLenFeature([400], tf.float32),
        'top_context': tf.FixedLenFeature([400], tf.int64),
        'number_of_tracks_in_sessions': tf.FixedLenFeature([400], tf.int64),
        'session_start_time': tf.FixedLenFeature([400], tf.int64)
    }

    sample = tf.parse_single_example(data_record, features)

    for dyntype in ['hist_avgs','histavgs_skip','histavgs_listen','histavgs_feedback']:
        sample[dyntype] = tf.sparse.to_dense(sample[dyntype])
        sample[dyntype] = tf.reshape(sample[dyntype], sample['hist_avgs_shape'])

    for v in features.keys():# ['day_hist','hour_hist','minute_hist','device_hist','day_now','hour_now','minute_now','device_now','hist_avgs','target_avg','target_skip']:
        if 'mask' not in v and 'time_since_last_session' not in v and 'session_start_time' not in v:
            sample[v] = tf.cast(sample[v], tf.int32)

    sample['mask'] = tf.cast(sample['mask'], tf.float32)
    sample['mask_split'] = tf.cast(sample['mask_split'], tf.float32)
    sample['mask_split_above'] = tf.cast(sample['mask_split_above'], tf.float32)

    #sample['time_since_last_session'] = tf.cast(sample['time_since_last_session'], tf.float32)

    sample['number_of_tracks_in_sessions'] = tf.cast(sample['number_of_tracks_in_sessions'], tf.float32)
    #sample['session_start_time'] = tf.cast(sample['session_start_time'], tf.float32)


    sample['user'] = tf.squeeze(sample['user'], -1)

    return tuple([sample[f] for f in ['day_hist','hour_hist','minute_hist',
                                      'device_hist','hist_avgs','mask','mask_split','user',
                                      'histavgs_skip','histavgs_listen','histavgs_feedback',
                                      'mask_split_above', 'time_since_last_session', 'top_context', 'number_of_tracks_in_sessions', 'session_start_time']]) #


def make_dataset_generator_onerow(sess, handle, pargs, dataset_paths, is_test):
    maxlen = pargs['max_length']

    output_t = tuple([tf.int32, tf.int32, tf.int32, tf.int32,  # history list: day, hour, minute, device
                      tf.int32,  # history list: average session vector
                      tf.float32,  # mask
                      tf.float32,  #mask split
                      tf.int32,  # user
                      tf.int32,
                      tf.int32,
                      tf.int32,
                      tf.float32,  # mask split above
                      tf.float32, # time since last sess
                      tf.int32, # top context
                      tf.float32,
                      tf.int64
                      ])

    filenames = dataset_paths
    print(len(filenames), 'tfrecord files')

    output_s = tuple(
        [tf.TensorShape([None, maxlen]), tf.TensorShape([None, maxlen]), tf.TensorShape([None, maxlen]), tf.TensorShape([None, maxlen]),
         tf.TensorShape([None, maxlen, 10]),
         tf.TensorShape([None, maxlen]),
         tf.TensorShape([None, maxlen]),
         tf.TensorShape([None, ]),
         tf.TensorShape([None, maxlen, 10]),
         tf.TensorShape([None, maxlen, 10]),
         tf.TensorShape([None, maxlen, 10]),
         tf.TensorShape([None, maxlen]),
         tf.TensorShape([None, maxlen]),
         tf.TensorShape([None, maxlen]),
         tf.TensorShape([None, maxlen]),
         tf.TensorShape([None, maxlen])
         ])

    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if not is_test:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(1000)
    dataset = dataset.flat_map(tf.data.TFRecordDataset)
    dataset = dataset.map(extract_fn_onerow, num_parallel_calls=3)
    if not is_test:
        dataset = dataset.shuffle(1000)

    if not is_test:
        dataset = dataset.batch(pargs["batch_size"])
    else:
        dataset = dataset.batch(500)

    if not is_test:
        dataset = dataset.prefetch(1)
    else:
        dataset = dataset.prefetch(10)
    iterator = dataset.make_initializable_iterator() #tf.compat.v1.data.make_initializable_iterator(dataset) #

    if handle is not None:
        generic_iter = tf.data.Iterator.from_string_handle(handle, output_t, output_s)
        specific_handle = sess.run(iterator.string_handle())
    else:
        generic_iter = None
        specific_handle = None

    return specific_handle, iterator, generic_iter
