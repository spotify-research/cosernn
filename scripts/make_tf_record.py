import numpy as np
import os
import tensorflow as tf

def make_record(num_samples=100, repeats=10):
    try:
        os.mkdir("records")
    except Exception as e:
        pass
    for repeat in range(repeats):
        writer = tf.python_io.TFRecordWriter("records/random_record" + str(repeat))

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=values))
            return f

        def create_float_feature(values):
            f = tf.train.Feature(float_list=tf.train.FloatList(value=values))
            return f

        def create_numpy_matrix(size, above_val=0.5, type=np.int64, add=1):
            v = ((np.random.random(size)>above_val).astype(int) + add).astype(type)
            return v

        for i in range(num_samples):
            features = {
                'day_hist': create_int_feature(create_numpy_matrix(400)), #tf.FixedLenFeature([400], tf.int64),
                'hour_hist': create_int_feature(create_numpy_matrix(400)), #tf.FixedLenFeature([400], tf.int64),
                'minute_hist': create_int_feature(create_numpy_matrix(400)), #tf.FixedLenFeature([400], tf.int64),
                'device_hist': create_int_feature(create_numpy_matrix(400)), #tf.FixedLenFeature([400], tf.int64),

                'hist_avgs': create_int_feature(create_numpy_matrix((400,10)).flatten()), #tf.VarLenFeature(tf.int64),
                'histavgs_feedback': create_int_feature(create_numpy_matrix((400,10)).flatten()), #tf.VarLenFeature(tf.int64),
                'histavgs_skip': create_int_feature(create_numpy_matrix((400,10)).flatten()), #tf.VarLenFeature(tf.int64),
                'histavgs_listen':create_int_feature(create_numpy_matrix((400,10)).flatten()), #tf.VarLenFeature(tf.int64),

                'hist_avgs_shape': create_int_feature(create_numpy_matrix((400,10)).shape), #tf.FixedLenFeature([2], tf.int64),

                'user': create_int_feature(create_numpy_matrix(1)),#tf.FixedLenFeature([1], tf.int64),
                'mask': create_int_feature(create_numpy_matrix(400, above_val=-1, add=0)), #tf.FixedLenFeature([400], tf.int64),
                'mask_split': create_int_feature(create_numpy_matrix(400, above_val=-1, add=0)), #tf.FixedLenFeature([400], tf.int64),
                'mask_split_above': create_int_feature(create_numpy_matrix(400, above_val=-1, add=0)), #tf.FixedLenFeature([400], tf.int64),

                'time_since_last_session': create_float_feature(create_numpy_matrix(400, type=np.float32)), #tf.FixedLenFeature([400], tf.float32),
                'top_context': create_int_feature(create_numpy_matrix(400)), #tf.FixedLenFeature([400], tf.int64),
                'number_of_tracks_in_sessions': create_int_feature(create_numpy_matrix(400)+10), #tf.FixedLenFeature([400], tf.int64),
                'session_start_time': create_int_feature(create_numpy_matrix(400)) #tf.FixedLenFeature([400], tf.int64)
            }

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())

        writer.close()

make_record()
