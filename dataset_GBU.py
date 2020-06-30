from sklearn import preprocessing
import h5py
import numpy as np
import scipy.io as sio
import tensorflow as tf
import copy

def map_label(label,classes):
    mapped_label=np.zeros(label.shape,dtype=np.int64)
    label=label.numpy()
    classes=classes.numpy()
    for i in range(classes.shape[0]):
        mapped_label[label == classes[i]] = i
    return tf.convert_to_tensor(mapped_label)

#导入数据，并进行处理
class DATA_LOADER(object):
    def __init__(self,opt):
        if opt.dataset=='imageNet1K':
            self.read_matimagenet(opt)
        else:
            self.read_matdataset(opt)
        self.index_in_epoch=0
        self.epochs_completed=0
        self.feature_dim=self.train_feature.shape[1]
        self.att_dim=self.attribute.shape[1]
        self.text_dim=self.att_dim
        self.tr_cls_centroid=np.zeros([self.seenclasses.shape[0], self.feature_dim], np.float32)
        for i in range(self.seenclasses.shape[0]):
            self.tr_cls_centroid[i] = np.mean(self.train_feature[self.train_label == i].numpy(), axis=0)

    def read_matimagenet(self,opt):
        if opt.preprocessing:
            print('MinMaxScaler...')
            scaler = preprocessing.MinMaxScaler()
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = scaler.fit_transform(np.array(matcontent['features']))
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = scaler.transform(np.array(matcontent['features_val']))
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()
            matcontent = h5py.File('/BS/xian/work/data/imageNet21K/extract_res/res101_1crop_2hops_t.mat', 'r')
            feature_unseen = scaler.transform(np.array(matcontent['features']))
            label_unseen = np.array(matcontent['labels']).astype(int).squeeze() - 1
            matcontent.close()
        else:
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = np.array(matcontent['features'])
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = np.array(matcontent['features_val'])
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()

        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + ".mat")
        self.attribute = tf.convert_to_tensor(matcontent['w2v'],dtype=tf.float32)
        self.train_feature = tf.convert_to_tensor(feature,dtype=tf.float32)
        self.train_label = tf.convert_to_tensor(label,dtype=tf.int64)
        self.test_seen_feature = tf.convert_to_tensor(feature_val,dtype=tf.float32)
        self.test_seen_label = tf.convert_to_tensor(label_val,dtype=tf.int64)
        self.test_unseen_feature = tf.convert_to_tensor(feature_unseen,dtype=tf.float32)
        self.test_unseen_label = tf.convert_to_tensor(label_unseen,dtype=tf.int64)
        self.ntrain = self.train_feature.shape()[0]
        self.seenclasses =tf.convert_to_tensor(np.unique(self.train_label.numpy()))
        self.unseenclasses = tf.convert_to_tensor(np.unique(self.test_unseen_label.numpy()))
        self.train_class = tf.convert_to_tensor(np.unique(self.train_label.numpy()))
        self.ntrain_class = self.seenclasses.shape(0)
        self.ntest_class = self.unseenclasses.shape(0)

    def read_matdataset(self,opt):
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        feature = matcontent['features'].T#AWA是(30475, 2048)
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1#(19832,)
        train_loc = matcontent['train_loc'].squeeze() - 1#(16864,)
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1#(7926,)
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1#(4958,)
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1#(5685,)

        self.attribute=tf.convert_to_tensor(matcontent['att'].T,dtype=tf.float32)#(50, 85)
        if not opt.validation:
            scaler = preprocessing.MinMaxScaler()
            _train_feature = scaler.fit_transform(feature[trainval_loc])
            _test_seen_feature = scaler.transform(feature[test_seen_loc])
            _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
            self.train_feature = tf.convert_to_tensor(_train_feature,dtype=tf.float32)
            mx=tf.convert_to_tensor(self.train_feature.numpy().max())
            self.train_feature=1/mx*self.train_feature
            self.train_label=tf.convert_to_tensor(label[trainval_loc],dtype=tf.int64)
            self.test_unseen_feature=tf.convert_to_tensor(_test_unseen_feature,dtype=tf.float32)
            self.test_unseen_feature=(1/mx)*self.test_unseen_feature
            self.test_unseen_label=tf.convert_to_tensor(label[test_unseen_loc],dtype=tf.int64)
            self.test_seen_feature=tf.convert_to_tensor(_test_seen_feature,dtype=tf.float32)
            self.test_seen_feature=self.test_seen_feature*(1/mx)
            self.test_seen_label=tf.convert_to_tensor(label[test_seen_loc],dtype=tf.int64)
        else:
            self.train_feature = tf.convert_to_tensor(feature[train_loc],dtype=tf.float32)
            self.train_label = tf.convert_to_tensor(label[train_loc],dtype=tf.int64)
            self.test_unseen_feature = tf.convert_to_tensor(feature[val_unseen_loc],dtype=tf.float32)
            self.test_unseen_label = tf.convert_to_tensor(label[val_unseen_loc],dtype=tf.int64)

        self.seenclasses = tf.convert_to_tensor(np.unique(self.train_label.numpy()))
        self.unseenclasses = tf.convert_to_tensor(np.unique(self.test_unseen_label.numpy()))
        self.ntrain = self.train_feature.shape[0]
        self.ntrain_class = self.seenclasses.shape[0]
        self.ntest_class = self.unseenclasses.shape[0]
        self.train_class = copy.deepcopy(self.seenclasses)
        self.allclasses=tf.range(0,self.ntrain_class + self.ntest_class,dtype=tf.int64)

        self.train_label = map_label(self.train_label, self.seenclasses)#此时训练集合为0-39，不再按照之前的类名了
        self.test_unseen_label = map_label(self.test_unseen_label, self.unseenclasses)#0-9
        self.test_seen_label = map_label(self.test_seen_label, self.seenclasses)#0-39
        self.train_att = self.attribute.numpy()[self.seenclasses.numpy()]
        self.test_att  = self.attribute.numpy()[self.unseenclasses.numpy()]

class FeatDataLayer(object):
    def __init__(self, label, feat_data,  opt):
        assert len(label)==feat_data.shape[0]
        self._opt=opt
        self._feat_data = feat_data
        self._label = label
        self._shuffle_roidb_inds()
        self._epoch = 0

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._label)))
        # self._perm = np.arange(len(self._roidb))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + self._opt.batchsize >= len(self._label):
            self._shuffle_roidb_inds()
            self._epoch += 1

        db_inds = self._perm[self._cur:self._cur + self._opt.batchsize]
        self._cur += self._opt.batchsize
        return db_inds

    def forward(self):
        new_epoch=False
        if self._cur+self._opt.batchsize>=len(self._label):
            self._shuffle_roidb_inds()
            self._epoch += 1
            new_epoch = True
        db_inds = self._perm[self._cur:self._cur + self._opt.batchsize]
        self._cur += self._opt.batchsize

        minibatch_feat = np.array([self._feat_data[i] for i in db_inds])
        minibatch_label = np.array([self._label[i] for i in db_inds])
        blobs = {'data': minibatch_feat, 'labels': minibatch_label, 'newEpoch': new_epoch, 'idx': db_inds}
        return blobs

    def get_whole_data(self):
        blobs={'data': self._feat_data, 'labels': self._label}
        return blobs
