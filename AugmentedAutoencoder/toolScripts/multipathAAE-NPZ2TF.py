'''
    Support Script to convert npz to tfrecord
'''
import numpy as np
import tensorflow as tf
import cv2
import progressbar
import sys
import os
import glob

try:
    range = xrange
except NameError:
    pass    # in order to be compatible with python2

try:
    input = raw_input
except NameError:
    pass

def yes_or_no(text):
    answer = input(text + "(y/n): ").lower().strip()
    print ("")
    if answer[0] == 'y':
        return True
    return False

def deserializer(dataset):
    keys_to_features = {'train_x':tf.FixedLenFeature([], tf.string),
                        'train_y':tf.FixedLenFeature([], tf.string),
                        'mask':tf.FixedLenFeature([], tf.string)}

    parsed_features = tf.parse_single_example(dataset, keys_to_features)
    train_x = tf.reshape(tf.decode_raw(parsed_features['train_x'], tf.uint8), (128,128,3))
    mask_x = tf.reshape(tf.decode_raw(parsed_features['mask'], tf.uint8), (128,128))
    train_y = tf.reshape(tf.decode_raw(parsed_features['train_y'], tf.uint8), (128,128,3))

    return (train_x, mask_x, train_y)

def convertNPZ2TFRecord(npzpath, tfrecordpath):
    with np.load(npzpath) as npzdata:
        # print all keys
        print (npzdata.files)
        #for k, v in npzdata.iteritems():
        #    print ('key {} has data shape {} and type {}'.format(k, v.shape, v.dtype))
        
        # try to output a sample to look
        # debug mode, only output 10 data in order to test serialization
        data_size = npzdata['bgr_x'].shape[0]
        # data_size = 10  # debug

        # progressbar
        widgets = ['Processing: ', progressbar.Percentage(),
         ' ', progressbar.Bar(),
         ' ', progressbar.Counter(), ' / %s' % data_size,
         ' ', progressbar.ETA(), ' ']
        bar = progressbar.ProgressBar(maxval=data_size,widgets=widgets)

        # create tf record
        generateFilePath = tfrecordpath

        # check if file exist
        create_file = True
        if os.path.exists(tfrecordpath):
            create_file = False
            if yes_or_no('file %s exists, do you want to replace it?' % generateFilePath):
                create_file = True
        
        if create_file:
            writer = tf.python_io.TFRecordWriter(generateFilePath)
            train_xs = npzdata['bgr_x']
            train_ys = npzdata['bgr_y']
            train_maskes = npzdata['mask_x']
            cv2.imwrite('D:/test.jpg',train_xs[0])
            bar.start()

            for i in range(data_size):      # if it's python2, better change to xrange
                train_x = train_xs[i]
                train_y = train_ys[i]
                mask_x = train_maskes[i]

                ser_train_x = train_x.tostring()
                ser_train_y = train_y.tostring()
                ser_mask_x = mask_x.tostring()

                feature = {}
                feature['train_x'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[ser_train_x]))
                feature['train_y'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[ser_train_y]))
                feature['mask'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[ser_mask_x]))

                example = tf.train.Example(features=tf.train.Features(feature=feature))
                serialized = example.SerializeToString()
                writer.write(serialized)

                bar.update(i)
        
            writer.close()
            bar.finish()
            print('converted file output to %s' % generateFilePath)
        
        check_file = create_file
        if not create_file:
            if yes_or_no("Do you want to check the file %s?" % generateFilePath):
               check_file = True 

        # tester
        if check_file:
            batch_size = min(max(int(data_size / 20), 1), 1000)

            dataset = tf.data.TFRecordDataset(generateFilePath)
            dataset = dataset.map(deserializer)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(1)
            iterator = dataset.make_one_shot_iterator()
            print ('checking data')
            total_sample = 0
            bar.start()
            with tf.Session() as sess:
                for i in range(0, data_size, batch_size):
                    features = sess.run(iterator.get_next())
                    # convert to np and output
                    # check nan and inf
                    if np.any(np.isnan(features[0])):
                        print ('nan detected in batch number:%d, train_x' % i)
                    if np.any(np.isnan(features[1])):
                        print ('nan detected in batch number:%d, mask' % i)
                    if np.any(np.isnan(features[2])):
                        print ('nan detected in batch number:%d, train_y' % i)
                    if np.any(np.isinf(features[0])):
                        print ('inf detected in batch number:%d, train_x' % i)
                    if np.any(np.isinf(features[1])):
                        print ('inf detected in batch number:%d, mask' % i)
                    if np.any(np.isinf(features[2])):
                        print ('inf detected in batch number:%d, train_y' % i)
                
                    total_sample = i + batch_size
                    bar.update(total_sample)
            bar.finish()
            print ('Total %d samples checked' % total_sample)


if __name__ == '__main__':
    argc = len(sys.argv)
    if argc > 3 or argc <= 1:
        print ('Two parameters are required, the first one is for input .npz files, the second one is for output path')
        print ('The output path is optional.')
        exit()
    
    npzPaths = glob.glob(sys.argv[1])
    for npzfile in npzPaths:
        print ('start to convert file:%s' % npzfile)
        output_path = os.path.dirname(npzfile)
        if argc == 3:
            output_path = sys.argv[2]
        file_name = os.path.split(npzfile)[-1]
        file_name = file_name[0:file_name.rfind('.')]
        output_file_path = os.path.join(output_path, file_name + '.tfrecord')
        print ('converting to %s' % output_file_path)
        convertNPZ2TFRecord(npzfile, output_file_path)
