from tensorflow import keras
import tensorflow as tf
import numpy as np

class LINEAR_LOGSOFTMAX(keras.Model):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc=keras.layers.Dense(nclass,kernel_initializer=keras.initializers.GlorotNormal(),bias_initializer='zeros')
        self.logic=keras.layers.Softmax(axis=1)
    def call(self, inputs, training=None, mask=None):
        out=tf.math.log(self.logic(self.fc(inputs)))
        return out

class CLASSIFIER:
    # train_Y is interger
    def __init__(self, _train_X, _train_Y, data_loader, _nclass, _lr=0.001, _beta1=0.5, _nepoch=20,
                 _batch_size=100, generalized=True, MCA=True):
        #_train_X和_train_Y最好是numpy
        self.train_X =  _train_X.numpy()
        self.train_Y = _train_Y.numpy()
        self.test_seen_feature = data_loader.test_seen_feature
        self.test_seen_label = data_loader.test_seen_label
        self.test_unseen_feature = data_loader.test_unseen_feature
        self.test_unseen_label = data_loader.test_unseen_label
        self.seenclasses = data_loader.seenclasses
        self.unseenclasses = data_loader.unseenclasses
        self.ntrain_class = data_loader.ntrain_class
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = _nclass
        self.input_dim = _train_X.shape[1]
        self.MCA = MCA
        self.model = LINEAR_LOGSOFTMAX(self.input_dim, self.nclass)
        self.criterion=keras.losses.CategoricalCrossentropy()

        self.input=tf.random.normal(shape=(_batch_size, self.input_dim),dtype=tf.float32)
        self.output=tf.random.normal(shape=(_batch_size,),dtype=tf.float64)

        self.lr=_lr
        self.beta1 = _beta1
        # setup optimizer
        self.optimizer=keras.optimizers.Adam(lr=_lr, beta_1=_beta1,beta_2=0.999)

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.shape[0]
        if generalized:
            self.acc_seen, self.acc_unseen, self.H = self.fit()

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm=tf.random.shuffle(tf.range(self.ntrain)).numpy()
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            # shuffle the data
            perm=tf.random.shuffle(tf.range(self.ntrain)).numpy()
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            #print(start, end)
            if rest_num_examples > 0:
                return tf.concat((tf.convert_to_tensor(X_rest_part),tf.convert_to_tensor(X_new_part)),0),tf.concat((tf.convert_to_tensor(Y_rest_part),tf.convert_to_tensor(Y_new_part)),0)
            else:
                return tf.convert_to_tensor(X_new_part),tf.convert_to_tensor(Y_new_part)
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            #print(start, end)
            # from index start to index end-1
            return tf.convert_to_tensor(self.train_X[start:end]), tf.convert_to_tensor(self.train_Y[start:end])

    def eval_MCA(self, preds, y):
        '''
        :param preds: numpy
        :param y: numpy
        :return: numpy
        '''
        cls_label = np.unique(y)
        acc = list()
        for i in cls_label:
            acc.append((preds[y == i] == i).mean())
        return np.asarray(acc).mean()

    def val_gzsl(self, test_X, test_label):
        start=0
        ntest=test_X.shape[0]
        predicted_label=tf.random.normal(shape=test_label.shape,dtype=tf.float64).numpy()
        for i in range(0, ntest, self.batch_size):
            end=min(ntest, start+self.batch_size)
            output=self.model(test_X[start:end])
            predicted_label[start:end]=np.argmax(output.numpy(),axis=1)
            start=end
        if self.MCA:
            acc=self.eval_MCA(predicted_label, test_label.numpy())
        else:
            acc = (predicted_label == test_label.numpy()).mean()
        return acc

    def fit(self):
        best_H = 0
        best_seen = 0
        best_unseen = 0
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):
                batch_input, batch_label = self.next_batch(self.batch_size)
                self.input=batch_input
                self.label=tf.one_hot(batch_label,depth=self.nclass)

                with tf.GradientTape() as tape:
                    output=self.model(self.input)
                    loss = self.criterion(output, self.label)
                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                # print('Training classifier loss= ', loss.data[0])
            acc_seen = 0
            acc_unseen = 0
            acc_seen = self.val_gzsl(self.test_seen_feature, self.test_seen_label)
            acc_unseen = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label + self.ntrain_class)
            H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
            # print('acc_seen=%.4f, acc_unseen=%.4f, h=%.4f' % (acc_seen, acc_unseen, H))
            if H > best_H:
                best_seen = acc_seen
                best_unseen = acc_unseen
                best_H = H
        return best_seen * 100, best_unseen * 100, best_H * 100