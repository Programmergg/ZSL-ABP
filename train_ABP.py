import argparse
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import classifier
from time import gmtime, strftime
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from dataset_GBU import DATA_LOADER,FeatDataLayer

parser=argparse.ArgumentParser('This program is to rewrite ABP_model with tensorflow2.0')
parser.add_argument('--dataset',default='AWA',help='dataset: CUB, AWA, AWA2, SUN')
parser.add_argument('--dataroot',default='F:\\data\\ZSLdata',help='path to dataset')
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--image_embedding', default='res101', type=str)
parser.add_argument('--class_embedding', default='att', type=str)

parser.add_argument('--nepoch', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate to train generater')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight_decay')
parser.add_argument('--batchsize', type=int, default=64, help='input batch size')
parser.add_argument('--nSample', type=int, default=300, help='number features to generate per class')

parser.add_argument('--resume',  type=str, help='the model to resume')
parser.add_argument('--disp_interval', type=int, default=20)
parser.add_argument('--save_interval', type=int, default=10000)
parser.add_argument('--evl_interval',  type=int, default=60)
parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--latent_dim', type=int, default=10, help='dimention of latent z')
parser.add_argument('--gh_dim',     type=int, default=4096, help='dimention of hidden layer in generator')
parser.add_argument('--latent_var', type=float, default=1, help='variance of prior distribution z')

parser.add_argument('--sigma',   type=float, default=0.1, help='variance of random noise')
parser.add_argument('--sigma_U', type=float, default=1,   help='variance of U_tau')
parser.add_argument('--langevin_s', type=float, default=0.1, help='s in langevin sampling')
parser.add_argument('--langevin_step', type=int, default=5, help='langevin step in each iteration')

parser.add_argument('--Knn', type=int, default=20, help='K value')
parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
opt = parser.parse_args()

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)

np.random.seed(opt.manualSeed)
random.seed(opt.manualSeed)
tf.random.set_seed(opt.manualSeed)
print('Running parameters:')
print(json.dumps(vars(opt), indent=4, separators=(',', ': ')))

class Conditional_Generator(keras.Model):
    def __init__(self,opt):
        super(Conditional_Generator, self).__init__()
        self.main=keras.models.Sequential([keras.layers.Dense(opt.gh_dim,kernel_initializer=keras.initializers.RandomNormal(mean=0,stddev=0.02),bias_initializer='zeros'),
                                   keras.layers.LeakyReLU(0.2),
                                   keras.layers.Dense(opt.X_dim,kernel_initializer=keras.initializers.RandomNormal(mean=0,stddev=0.02),bias_initializer='zeros'),
                                   keras.layers.ReLU()])
    def call(self, c, z):
        input=tf.concat([z,c],axis=1)
        output=self.main(input)
        return output

class Result(object):
    def __init__(self):
        self.best_acc = 0.0
        self.best_iter = 0.0
        self.best_acc_S_T = 0.0
        self.best_acc_U_T = 0.0
        self.acc_list = []
        self.iter_list = []
        self.save_model = False

    def update(self, it, acc):
        self.acc_list += [acc]
        self.iter_list += [it]
        self.save_model = False
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_iter = it
            self.save_model = True

    def update_gzsl(self, it, acc_u, acc_s, H):
        self.acc_list += [H]
        self.iter_list += [it]
        self.save_model = False
        if H > self.best_acc:
            self.best_acc = H
            self.best_iter = it
            self.best_acc_U_T = acc_u
            self.best_acc_S_T = acc_s
            self.save_model = True

def getloss(pred, x, z, opt):
    loss=1/(2*opt.sigma**2) * tf.reduce_sum(tf.math.pow(x-pred,2))+1/2*tf.reduce_sum(tf.math.pow(z,2))
    loss/=x.shape[0]
    return loss

def log_print(s,log):
    print(s)
    with open(log, 'a') as f:
        f.write(s + '\n')

def synthesize_feature_test(netG, dataset, opt):
    gen_feat=tf.random.normal(shape=(dataset.ntest_class * opt.nSample, opt.X_dim))
    gen_label = np.zeros([0])
    for i in range(dataset.ntest_class):
        text_feat = np.tile(dataset.test_att[i].astype('float32'), (opt.nSample, 1))
        text_feat=tf.convert_to_tensor(text_feat)
        z=tf.random.normal(shape=(opt.nSample, opt.Z_dim))
        G_sample = netG(z, text_feat)
        G_sample=G_sample.numpy()
        gen_feat=gen_feat.numpy()
        gen_feat[i * opt.nSample:(i + 1) * opt.nSample] = G_sample
        gen_feat=tf.convert_to_tensor(gen_feat)
        gen_label = np.hstack((gen_label, np.ones([opt.nSample]) * i))
    return gen_feat,tf.convert_to_tensor(gen_label,dtype=tf.int64)

def eval_MCA(preds, y):
    cls_label = np.unique(y)
    acc = list()
    for i in cls_label:
        acc.append((preds[y == i] == i).mean())
    return np.asarray(acc).mean()

def eval_zsl_knn(gen_feat, gen_label, dataset):
    # cosince predict K-nearest Neighbor
    n_test_sample = dataset.test_unseen_feature.shape[0]
    sim = cosine_similarity(dataset.test_unseen_feature, gen_feat)
    # only count first K nearest neighbor
    idx_mat=np.argsort(-1 * sim, axis=1)[:, 0:opt.Knn]
    label_mat = gen_label[idx_mat.flatten()].reshape((n_test_sample, -1))
    preds = np.zeros(n_test_sample)
    for i in range(n_test_sample):
        label_count = Counter(label_mat[i]).most_common(1)
        preds[i] = label_count[0][0]
    acc = eval_MCA(preds, dataset.test_unseen_label.numpy()) * 100
    return acc
def train():
    dataset=DATA_LOADER(opt)
    opt.C_dim = dataset.att_dim
    opt.X_dim = dataset.feature_dim
    opt.Z_dim = opt.latent_dim
    opt.y_dim = dataset.ntrain_class
    opt.niter = int(dataset.ntrain/opt.batchsize) * opt.nepoch#309000

    data_layer=FeatDataLayer(dataset.train_label.numpy(), dataset.train_feature.numpy(), opt)
    result_zsl_knn = Result()
    result_gzsl_soft = Result()
    netG = Conditional_Generator(opt)
    print('Conditional_Generator:',netG)

    train_z=tf.random.normal(mean=0,stddev=opt.latent_var,shape=(len(dataset.train_feature), opt.Z_dim))
    out_dir='out/{}/nSample-{}_nZ-{}_sigma-{}_langevin_s-{}_step-{}'.format(opt.dataset, opt.nSample, opt.Z_dim,
                                                                              opt.sigma, opt.langevin_s, opt.langevin_step)
    os.makedirs(out_dir, exist_ok=True)
    print("The output dictionary is {}".format(out_dir))
    log_dir = out_dir + '/log_{}.txt'.format(opt.dataset)
    with open(log_dir, 'w') as f:
        f.write('Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')

    start_step=0
    #这里复用不想写
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    optimizerG=keras.optimizers.Adam(lr=opt.lr,decay=opt.weight_decay)

    # range(start_step, opt.niter+1)
    for it in range(start_step, opt.niter+1):
        blobs = data_layer.forward()
        feat_data = blobs['data']  # image data(64, 2048)
        labels = blobs['labels'].astype(int)  # class labels(64,)
        idx = blobs['idx'].astype(int)#(64,)

        C=np.array([dataset.train_att[i,:] for i in labels])
        C=tf.convert_to_tensor(C,dtype=tf.float32)#(64,85)
        X=tf.convert_to_tensor(feat_data)#(64,2048)
        Z=tf.convert_to_tensor(train_z.numpy()[idx])#(64,10)
        optimizer_z=keras.optimizers.Adam(lr=opt.lr,decay=opt.weight_decay)

        # Alternatingly update weights w and infer latent_batch z
        for em_step in range(2):  # EM_STEP
            #UPDATE W
            for _ in range(1):
                with tf.GradientTape() as tape:
                    pred = netG(Z, C)
                    loss = getloss(pred, X, Z, opt)
                grads=tape.gradient(loss,netG.trainable_variables)
                #进行梯度裁剪，防止梯度爆炸
                for i,grad in enumerate(grads):
                    grads[i]=tf.clip_by_norm(grad,1)
                optimizerG.apply_gradients(zip(grads, netG.trainable_variables))
            #infer z
            for _ in range(opt.langevin_step):
                U_tau = tf.random.normal(mean=0, stddev=opt.sigma_U, shape=Z.shape)
                with tf.GradientTape() as tape:
                    pred=netG(Z,C)
                    loss = getloss(pred, X, Z, opt)
                    loss = opt.langevin_s * 2 / 2 * loss
                grads = tape.gradient(loss, netG.trainable_variables)
                Z=tf.clip_by_norm(Z,1)
                optimizer_z.apply_gradients(zip(grads, netG.trainable_variables))
                if it < opt.niter/3:
                    Z += opt.langevin_s *U_tau
        #update Z
        train_z=train_z.numpy()
        train_z[idx,] = Z
        # print(train_z[idx,].shape)
        train_z=tf.convert_to_tensor(train_z)
        if it % opt.disp_interval == 0 and it:
            log_text = 'Iter-[{}/{}]; loss: {:.3f}'.format(it, opt.niter, loss)
            log_print(log_text, log_dir)

        if it % opt.evl_interval == 0 and it:
            gen_feat, gen_label = synthesize_feature_test(netG, dataset, opt)#(3000,2048)(3000,)
            """ZSL"""
            acc = eval_zsl_knn(gen_feat.numpy(), gen_label.numpy(), dataset)
            result_zsl_knn.update(it, acc)
            log_print("{}nn Classifer: ".format(opt.Knn), log_dir)
            log_print("Accuracy is {:.2f}%, Best_acc [{:.2f}% | Iter-{}]".format(acc, result_zsl_knn.best_acc,
                                                                             result_zsl_knn.best_iter), log_dir)
            # """GZSL"""
            # # note test label need be shift with offset ntrain_class
            # train_X=tf.concat((dataset.train_feature, gen_feat), 0)#(22832, 2048)
            # train_Y=tf.concat((dataset.train_label, gen_label+dataset.ntrain_class), 0)#(22832，)
            # cls = classifier.CLASSIFIER(train_X, train_Y, dataset, dataset.ntrain_class + dataset.ntest_class,
            #                                   opt.classifier_lr, 0.5, 25, opt.nSample, True)
            # result_gzsl_soft.update_gzsl(it, cls.acc_unseen, cls.acc_seen, cls.H)
            # log_print("GZSL Softmax:", log_dir)
            # log_print("U->T {:.2f}%  S->T {:.2f}%  H {:.2f}%  Best_H [{:.2f}% {:.2f}% {:.2f}% | Iter-{}]".format(
            #     cls.acc_unseen, cls.acc_seen, cls.H,  result_gzsl_soft.best_acc_U_T, result_gzsl_soft.best_acc_S_T,
            #     result_gzsl_soft.best_acc, result_gzsl_soft.best_iter), log_dir)
train()