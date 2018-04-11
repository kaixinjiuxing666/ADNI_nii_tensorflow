#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import tensorlayer as tl
import numpy as np
import os, time, model
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def distort_imgs(data):
    """ data augumentation """
    x1, y = data
    x1, y = tl.prepro.flip_axis_multi([x1, y],
                            axis=1, is_random=True) # left right
    x1, y = tl.prepro.elastic_transform_multi([x1, y],
                            alpha=720, sigma=24, is_random=True)
    x1, y = tl.prepro.rotation_multi([x1, y], rg=20,
                            is_random=True, fill_mode='constant') # nearest, constant
    x1, y = tl.prepro.shift_multi([x1, y], wrg=0.10,
                            hrg=0.10, is_random=True, fill_mode='constant')
    x1, y = tl.prepro.shear_multi([x1, y], 0.05,
                            is_random=True, fill_mode='constant')
    x1, y = tl.prepro.zoom_multi([x1, y],
                            zoom_range=[0.9, 1.1], is_random=True,
                            fill_mode='constant')
    return x1, y

def vis_imgs(X, y, path):
    """ show one slice """
    if y.ndim == 2:
        y = y[:,:,np.newaxis]
    assert X.ndim == 3
    tl.vis.save_images(np.asarray([X[:,:,0,np.newaxis],y]),size=(1, 2),image_path=path)

def vis_imgs2(X, y_, y, path):
    """ show one slice with target """
    if y.ndim == 2:
        y = y[:,:,np.newaxis]
    if y_.ndim == 2:
        y_ = y_[:,:,np.newaxis]
    assert X.ndim == 3
    tl.vis.save_images(np.asarray([X[:,:,0,np.newaxis], y_, y]),size=(1, 3),image_path=path)

def main(task='all'):
    ## Create folder to save trained model and result images
    save_dir = "checkpoint"
    tl.files.exists_or_mkdir(save_dir)
    tl.files.exists_or_mkdir("samples/{}".format(task))

    ###======================== LOAD DATA ===================================###
    ## by importing this, you can load a training set and a validation set.
    # you will get X_train_input, X_train_target, X_dev_input and X_dev_target  
    dataset=np.load("192save.npz")
    X_train = dataset["arr_0"]
    y_train = dataset["arr_1"]#[:,:,:,np.newaxis]
    X_test = dataset["arr_2"]
    y_test = dataset["arr_3"]#[:,:,:,np.newaxis]

    if task == 'all':
        y_train = (y_train > 0).astype(int)
        y_test = (y_test > 0).astype(int)
    else:
        exit("Unknow task %s" % task)
    
    ###======================== HYPER-PARAMETERS ============================###
    batch_size = 10
    lr = 0.0001
    lr_decay = 0.5
    decay_every = 100
    beta1 = 0.9
    n_epoch = 5
    print_freq_step = 200

    ###======================== SHOW DATA ===================================###
    # show one slice
    X = np.asarray(X_train[105])
    y = np.asarray(y_train[105])
    print(X.shape, X.min(), X.max()) # (256, 256, 1) -0.352378 3.94396
    print(y.shape, y.min(), y.max()) # (256, 256, 1) 0 1
    nw, nh, nz = X.shape
    vis_imgs(X, y, 'samples/{}/_train_im.png'.format(task))

    with tf.device('/cpu:0'):
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        with tf.device('/gpu:0'): #<- remove it if you train on CPU or other GPU
            
            ###======================== DEFIINE MODEL =======================###  
            t_image = tf.placeholder('float32', [batch_size, nw, nh, 1], name='input_image')
            ## labels are either 0 or 1
            t_seg = tf.placeholder('float32', [batch_size, nw, nh, 1], name='target_segment')

            ## train inference
            #net = model.u_net(t_image, is_train=True, reuse=False, n_out=1)
            net = model.u_net_bn_192(t_image, is_train=True, reuse=False, batch_size=10, n_out=1)
            #net = model.u_net_bn_256(t_image, is_train=True, reuse=False, batch_size=20, n_out=1)
            
            ## test inference
            #net_test = model.u_net(t_image, is_train=False, reuse=True, n_out=1)
            net_test = model.u_net_bn_192(t_image, is_train=False, reuse=True, batch_size=10, n_out=1)
            #net_test = model.u_net_bn_256(t_image, is_train=False, reuse=True, batch_size=20, n_out=1)

            ###======================== DEFINE LOSS =========================###
            ## train losses
            out_seg = net.outputs
            dice_loss = 1 - tl.cost.dice_coe(out_seg, t_seg, axis=[0,1,2,3])#, 'jaccard', epsilon=1e-5)
            iou_loss = tl.cost.iou_coe(out_seg, t_seg, axis=[0,1,2,3])
            dice_hard = tl.cost.dice_hard_coe(out_seg, t_seg, axis=[0,1,2,3])
            loss = dice_loss
        
            ## test losses
            test_out_seg = net_test.outputs
            test_dice_loss = 1 - tl.cost.dice_coe(test_out_seg, t_seg, axis=[0,1,2,3])#, 'jaccard', epsilon=1e-5)
            test_iou_loss = tl.cost.iou_coe(test_out_seg, t_seg, axis=[0,1,2,3])
            test_dice_hard = tl.cost.dice_hard_coe(test_out_seg, t_seg, axis=[0,1,2,3])
            #cup_feature_guard
        ###======================== DEFINE TRAIN OPTS =======================###
        #t_vars = tl.layers.get_variables_with_name('u_net', True, True)
        with tf.device('/gpu:0'):
            with tf.variable_scope('learning_rate'):
                lr_v = tf.Variable(lr, trainable=False)
            train_op = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(loss, var_list=net.all_params)
            #u_net:var_list=t_vars------u_net_bn:var_list=net.all_params

        ###======================== LOAD MODEL ==============================###
        tl.layers.initialize_global_variables(sess)
        ## load existing model if possible
        tl.files.load_and_assign_npz(sess=sess, name=save_dir+'/u_net_{}.npz'.format(task), network=net)

        ###======================== TRAINING ================================###
    for epoch in range(0, n_epoch+1):
        epoch_time = time.time()
        ## update decay learning rate at the beginning of a epoch
        if epoch !=0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr * new_lr_decay))
            log = " ** new learning rate: %f" % (lr * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr))
            log = " ** init lr: %f  decay_every_epoch: %d, lr_decay: %f" % (lr, decay_every, lr_decay)
            print(log)

        total_dice, total_iou, total_dice_hard, n_batch = 0, 0, 0, 0
        for batch in tl.iterate.minibatches(inputs=X_train, targets=y_train,
                                    batch_size=batch_size, shuffle=True):
            images, labels = batch
            #print(images.shape)#(2,256,256,1)
            step_time = time.time()
            ## data augumentation for a batch of images and label maps synchronously.
            data = tl.prepro.threading_data([_ for _ in zip(images[:,:,:,0, np.newaxis], labels)],
                                             fn=distort_imgs)
                                          #fn=distort_imgs) # (10, 5, 240, 240, 1)10张图片，5个种类
            b_images = data[:,0:1,:,:,:]  # (10, 4, 240, 240, 1)
            b_labels = data[:,1,:,:,:]
            #print(b_images.shape)
            #print(b_labels.shape)
            b_images = b_images.transpose((0,2,3,1,4))#(10,240,240,4,1,1)
            b_images.shape = (batch_size, nw, nh, nz)
            #b_labels = b_labels.transpose((0,2,3,1,4))
            #b_labels.shape = (batch_size, nw, nh, nz)


            ## update network
            _, _dice, _iou, _diceh, out = sess.run([train_op,
                    dice_loss, iou_loss, dice_hard, net.outputs],
                    {t_image: b_images, t_seg: b_labels})
            total_dice += _dice; total_iou += _iou; total_dice_hard += _diceh
            n_batch += 1

            ## you can show the predition here:
            vis_imgs2(b_images[0], b_labels[0], out[0], "samples/{}/_tmp.png".format(task))
            exit()

            #if _dice == 1: # DEBUG
            #    print("DEBUG")
            #    vis_imgs2(b_images[0], b_labels[0], out[0], "samples/{}/_debug.png".format(task))

            if n_batch % print_freq_step == 0:
                print("Epoch %d step %d 1-dice: %f hard-dice: %f iou: %f took %fs (2d with distortion)"
                % (epoch, n_batch, _dice, _diceh, _iou, time.time()-step_time))

            ## check model fail
            if np.isnan(_dice):
                exit(" ** NaN loss found during training, stop training")
            if np.isnan(out).any():
                exit(" ** NaN found in output images during training, stop training")
        print(" ** Epoch [%d/%d] train 1-dice: %f hard-dice: %f iou: %f took %fs (2d with distortion)" %
              (epoch, n_epoch, total_dice/n_batch, total_dice_hard/n_batch, total_iou/n_batch, time.time()-epoch_time))

        ## save a predition of training set
        for i in range(batch_size):
            if np.max(b_images[i]) > 0:
                vis_imgs2(b_images[i], b_labels[i], out[i], "samples/{}/train_{}.png".format(task, epoch))
                break
            elif i == batch_size-1:
                vis_imgs2(b_images[i], b_labels[i], out[i], "samples/{}/train_{}.png".format(task, epoch))

        ###======================== EVALUATION ==========================###
        total_dice, total_iou, total_dice_hard, n_batch = 0, 0, 0, 0
        for batch in tl.iterate.minibatches(inputs=X_test, targets=y_test,
                                        batch_size=batch_size, shuffle=True):
            b_images, b_labels = batch
            #print(b_labels.shape)#(2,256,256,1)
            #b_labels.shape = (batch_size, nw, nh, nz)
            _dice, _iou, _diceh, out = sess.run([test_dice_loss,
                    test_iou_loss, test_dice_hard, net_test.outputs],
                    {t_image: b_images, t_seg: b_labels})
            total_dice += _dice; total_iou += _iou; total_dice_hard += _diceh
            n_batch += 1
        
        print(" **"+" "*17+"test 1-dice: %f hard-dice: %f iou: %f (2d no distortion)" %
              (total_dice/n_batch, total_dice_hard/n_batch, total_iou/n_batch))
        ## save a predition of test set
        for i in range(batch_size):
            if np.max(b_images[i]) > 0:
                vis_imgs2(b_images[i], b_labels[i], out[i], "samples/{}/test_{}.png".format(task, epoch))
                break
            elif i == batch_size-1:
                vis_imgs2(b_images[i], b_labels[i], out[i], "samples/{}/test_{}.png".format(task, epoch))

        ###======================== SAVE MODEL ==========================###
        tl.files.save_npz(net.all_params, name=save_dir+'/u_net_{}.npz'.format(task), sess=sess)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='all', help='all, necrotic, edema, enhance')

    args = parser.parse_args()

    main(args.task)
