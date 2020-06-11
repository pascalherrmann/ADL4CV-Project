"""Main script for training encoder. This script should be run
only after the stylegan's generator is well-trained"""

import os
import time
import sys
import tensorflow as tf
import numpy as np
import dnnlib
from dnnlib import EasyDict
import dnnlib.tflib as tflib
from training import misc
from perceptual_model import PerceptualModel
from utils.visualizer import fuse_images
from utils.visualizer import save_image
from utils.visualizer import adjust_pixel_range

import config

def process_reals(x, mirror_augment, drange_data, drange_net):
    with tf.name_scope('ProcessReals'):
        with tf.name_scope('DynamicRange'):
            x = tf.cast(x, tf.float32)
            x = misc.adjust_dynamic_range(x, drange_data, drange_net)
        if False: #mirror_augment: #todo
            with tf.name_scope('MirrorAugment'):
                s = tf.shape(x)
                mask = tf.random_uniform([s[0], 1, 1, 1], 0.0, 1.0)
                mask = tf.tile(mask, [1, s[1], s[2], s[3]])
                x = tf.where(mask < 0.5, x, tf.reverse(x, axis=[3]))
        return x


# new: extracting landmark
def parse_tfrecord_tf(record):
    features = tf.parse_single_example(record, features={
        'shape': tf.FixedLenFeature([3], tf.int64),
        'portrait': tf.FixedLenFeature([], tf.string),
        'landmark': tf.FixedLenFeature([], tf.string)})
    portrait = tf.decode_raw(features['portrait'], tf.uint8)
    landmark = tf.decode_raw(features['landmark'], tf.uint8)
    portrait = tf.reshape(portrait, (1, features['shape'][0], features['shape'][1], features['shape'][2]))
    landmark = tf.reshape(landmark, (1, features['shape'][0], features['shape'][1], features['shape'][2]))
    data = tf.concat((portrait, landmark), axis=0)
    return data


##
## todo
##
def get_train_data(sess, data_dir, submit_config, mode):
    if mode == 'train':
        shuffle = True; repeat = True; batch_size = submit_config.batch_size
    elif mode == 'test':
        shuffle = False; repeat = True; batch_size = submit_config.batch_size_test
    else:
        raise Exception("mode must in ['train', 'test'], but got {}" % mode)

    dset = tf.data.TFRecordDataset(data_dir)
    dset = dset.map(parse_tfrecord_tf, num_parallel_calls=16)

    if shuffle:
        bytes_per_item = np.prod([3*2, submit_config.image_size, submit_config.image_size]) * np.dtype('uint8').itemsize
        dset = dset.shuffle(((4096 << 20) - 1) // bytes_per_item + 1)
    if repeat:
        dset = dset.repeat()
    dset = dset.batch(batch_size)

    train_iterator = tf.data.Iterator.from_structure(dset.output_types, dset.output_shapes)
    training_init_op = train_iterator.make_initializer(dset)
    image_batch = train_iterator.get_next()
    sess.run(training_init_op)
    return image_batch


def test(E, Gs, real_test, submit_config):
    with tf.name_scope("Run"), tf.control_dependencies(None):
        with tf.device("/cpu:0"):
            in_split = tf.split(real_test, submit_config.num_gpus)
        out_split = []
        num_layers, latent_dim = Gs.components.synthesis.input_shape[1:3]
        for gpu in range(submit_config.num_gpus):
            with tf.device("/gpu:%d" % gpu):
                in_gpu = in_split[gpu]
                latent_w = E.get_output_for(in_gpu, phase=False)
                latent_wp = tf.reshape(latent_w, [in_gpu.shape[0], num_layers, latent_dim])
                fake_X_val = Gs.components.synthesis.get_output_for(latent_wp, randomize_noise=False)
                out_split.append(fake_X_val)

        with tf.device("/cpu:0"):
            out_expr = tf.concat(out_split, axis=0)

    return out_expr


def training_loop(
                  submit_config,
                  Encoder_args            = {},
                  E_opt_args              = {},
                  D_opt_args              = {},
                  E_loss_args             = EasyDict(),
                  D_loss_args             = {},
                  lr_args                 = EasyDict(),
                  tf_config               = {},
                  dataset_args            = EasyDict(),
                  decoder_pkl             = EasyDict(),
                  drange_data             = [0, 255],
                  drange_net              = [-1,1],   # Dynamic range used when feeding image data to the networks.
                  mirror_augment          = False,
                  resume_run_id           = config.ENCODER_PICKLE_DIR,     # Run ID or network pkl to resume training from, None = start from scratch.
                  resume_snapshot         = None,     # Snapshot index to resume training from, None = autodetect.
                  image_snapshot_ticks    = 1,        # How often to export image snapshots?
                  network_snapshot_ticks  = 4,       # How often to export network snapshots?
                  max_iters               = 150000):

    tflib.init_tf(tf_config)

    with tf.name_scope('input'):
        # das kommt aus dem feeddict!!!
        real_train = tf.placeholder(tf.float32, [submit_config.batch_size, 3, submit_config.image_size, submit_config.image_size], name='real_image_train')
        real_test = tf.placeholder(tf.float32, [submit_config.batch_size_test, 3, submit_config.image_size, submit_config.image_size], name='real_image_test')
        real_split = tf.split(real_train, num_or_size_splits=submit_config.num_gpus, axis=0)

        # placeholders for landmark inputs for the network (is set in feed dict)
        real_landmarks_train = tf.placeholder(tf.float32, [submit_config.batch_size, 3, submit_config.image_size, submit_config.image_size], name='real_landmarks_train')
        real_landmarks_test = tf.placeholder(tf.float32, [submit_config.batch_size_test, 3, submit_config.image_size, submit_config.image_size], name='real_landmarks_test')
        real_landmarks_split = tf.split(real_landmarks_train, num_or_size_splits=submit_config.num_gpus, axis=0)


    with tf.device('/gpu:0'):
        if resume_run_id is not None:
            network_pkl = misc.locate_network_pkl(resume_run_id, resume_snapshot)
            print('Loading networks from "%s"...' % network_pkl)
            E, G, D, Gs = misc.load_pkl(network_pkl)
            start = int(network_pkl.split('-')[-1].split('.')[0]) // submit_config.batch_size
            print('Start: ', start)
        else:
            print('Constructing networks...')
            G, _, Gs = misc.load_pkl(decoder_pkl.decoder_pkl)

            # Creating New Discriminator
            # as subclass of tflib.Network
            # specification in training.networks_stylegan.D_basic


            #
            # Here, the discriminator is initialized.
            # Creates an instnace of network, the architecture is defined in training.networks_stylegan.D_basic"
            # Also the arguments (num_channels, resolution, label_size) are passed to that function.
            #D = tflib.Network('D', num_channels=3, resolution=Gs.output_shape[3], label_size=0, func_name="training.networks_stylegan.D_basic")
            D = tflib.Network('D', size=128, filter=64, filter_max=512, num_layers=num_layers, phase=True, func_name='training.networks_encoder.Conditional_Discriminator')

            print("Creating NEW Discriminator!!!")
            num_layers = Gs.components.synthesis.input_shape[1]
            E = tflib.Network('E', size=submit_config.image_size, filter=64, filter_max=1024, num_layers=num_layers, phase=True, **Encoder_args)
            start = 0

    E.print_layers(); Gs.print_layers(); D.print_layers()

    global_step0 = tf.Variable(start, trainable=False, name='learning_rate_step')
    learning_rate = tf.train.exponential_decay(lr_args.learning_rate, global_step0, lr_args.decay_step,
                                               lr_args.decay_rate, staircase=lr_args.stair)
    add_global0 = global_step0.assign_add(1)

    E_opt = tflib.Optimizer(name='TrainE', learning_rate=learning_rate, **E_opt_args)
    D_opt = tflib.Optimizer(name='TrainD', learning_rate=learning_rate, **D_opt_args)

    E_loss_rec = 0.
    E_loss_adv = 0.
    D_loss_real = 0.
    D_loss_fake = 0.
    D_loss_grad = 0.
    for gpu in range(submit_config.num_gpus):
        print('build graph on gpu %s' % str(gpu))
        with tf.name_scope('GPU%d' % gpu), tf.device('/gpu:%d' % gpu):
            E_gpu = E if gpu == 0 else E.clone(E.name + '_shadow')
            D_gpu = D if gpu == 0 else D.clone(D.name + '_shadow')
            G_gpu = Gs if gpu == 0 else Gs.clone(Gs.name + '_shadow')
            perceptual_model = PerceptualModel(img_size=[E_loss_args.perceptual_img_size, E_loss_args.perceptual_img_size], multi_layers=False)
            real_gpu = process_reals(real_split[gpu], mirror_augment, drange_data, drange_net)

            # get landmarks gpu
            landmarks_gpu = process_reals(real_landmarks_split[gpu], mirror_augment, drange_data, drange_net)
            with tf.name_scope('E_loss'), tf.control_dependencies(None):

                #
                # get loss for encoder
                #
                E_loss, recon_loss, adv_loss = dnnlib.util.call_func_by_name(E=E_gpu, G=G_gpu, D=D_gpu, perceptual_model=perceptual_model, reals=real_gpu,real_landmarks=landmarks_gpu, **E_loss_args) #call loss function (loss_enocder)
                
                
                #
                # don't use recon-loss
                #
                #E_loss_rec += recon_loss
                

                E_loss_adv += adv_loss
            with tf.name_scope('D_loss'), tf.control_dependencies(None):
                #
                # get loss for discriminator
                #
                D_loss, loss_fake, loss_real, loss_gp = dnnlib.util.call_func_by_name(E=E_gpu, G=G_gpu, D=D_gpu, reals=real_gpu, real_landmarks=landmarks_gpu, **D_loss_args)
                D_loss_real += loss_real
                D_loss_fake += loss_fake
                D_loss_grad += loss_gp
            with tf.control_dependencies([add_global0]):
                E_opt.register_gradients(E_loss, E_gpu.trainables)
                D_opt.register_gradients(D_loss, D_gpu.trainables)

    E_loss_rec /= submit_config.num_gpus
    E_loss_adv /= submit_config.num_gpus
    D_loss_real /= submit_config.num_gpus
    D_loss_fake /= submit_config.num_gpus
    D_loss_grad /= submit_config.num_gpus

    E_train_op = E_opt.apply_updates()
    D_train_op = D_opt.apply_updates()

    print('building testing graph...')
    fake_X_val = test(E, Gs, real_test, submit_config)

    sess = tf.get_default_session()

    print('Getting training data...')
    image_batch_train = get_train_data(sess, data_dir=dataset_args.data_train, submit_config=submit_config, mode='train')
    image_batch_test = get_train_data(sess, data_dir=dataset_args.data_test, submit_config=submit_config, mode='test')

    summary_log = tf.summary.FileWriter(config.getGdrivePath())

    cur_nimg = start * submit_config.batch_size
    cur_tick = 0
    tick_start_nimg = cur_nimg
    start_time = time.time()

    init_fix = tf.initialize_variables(
        [global_step0],
        name='init_fix'
    )
    sess.run(init_fix)
    
    print('Optimization starts!!!')
    
    
    for it in range(start, max_iters):

        batch_images = sess.run(image_batch_train)

        #train
        portrait_images = batch_images[:,0,:,:,:]
        landmark_images = batch_images[:,1,:,:,:]


        feed_dict_1 = {real_train: portrait_images, real_landmarks_train: landmark_images} # todo: feed dict ändern.
        
        #print("portrait_images", portrait_images.shape)
        #print("landmark_images", landmark_images.shape)
        #print("E_train_op", E_train_op)
        #print("E_loss_rec", E_loss_rec)
        #print("E_loss_adv",E_loss_adv)
        
        # don't use E_loss_rec
        _, adv_ = sess.run([E_train_op, E_loss_adv], feed_dict_1) #todo: feed_dict ändern
        _, d_r_, d_f_, d_g_ = sess.run([D_train_op, D_loss_real, D_loss_fake, D_loss_grad], feed_dict_1)

        cur_nimg += submit_config.batch_size

        if it % 50 == 0:
            print('Iter: %06d recon_loss: %-6.4f adv_loss: %-6.4f d_r_loss: %-6.4f d_f_loss: %-6.4f d_reg: %-6.4f time:%-12s' % (
                it, -1, adv_, d_r_, d_f_, d_g_, dnnlib.util.format_time(time.time() - start_time)))
            sys.stdout.flush()
            tflib.autosummary.save_summaries(summary_log, it)
            
            
            
            
        if it % 500 == 0:
            batch_images_test = sess.run(image_batch_test)

            #test
            portrait_images_test = batch_images_test[:,0,:,:,:]
            landmark_images_test = batch_images_test[:,1,:,:,:]


            portrait_images_test = misc.adjust_dynamic_range(portrait_images_test.astype(np.float32), [0, 255], [-1., 1.])
            landmark_images_test = misc.adjust_dynamic_range(landmark_images_test.astype(np.float32), [0, 255], [-1., 1.])


            samples2 = sess.run(fake_X_val, feed_dict={real_test: portrait_images_test, real_landmarks_test: landmark_images_test})
            orin_recon = np.concatenate([portrait_images_test, landmark_images_test, samples2], axis=0) # bild: oben input, unten: reconstruct
            orin_recon = adjust_pixel_range(orin_recon)
            orin_recon = fuse_images(orin_recon, row=2, col=submit_config.batch_size_test)
            # save image results during training, first row is original images and the second row is reconstructed images
            save_image('%s/iter_%08d.png' % (submit_config.run_dir, cur_nimg), orin_recon)

            # save image to gdrive
            img_path = os.path.join(config.getGdrivePath(), 'images', ('iter_%08d.png' % (cur_nimg)))
            save_image(img_path, orin_recon)

        if cur_nimg >= tick_start_nimg + 65000:
            cur_tick += 1
            tick_start_nimg = cur_nimg



            if cur_tick % network_snapshot_ticks == 0:
                pkl = os.path.join(submit_config.run_dir, 'network-snapshot-%08d.pkl' % (cur_nimg))
                misc.save_pkl((E, G, D, Gs), pkl)
                
                # save network snapshot to gdrive
                pkl_drive = os.path.join(config.getGdrivePath(), 'snapshots', 'network-snapshot-%08d.pkl' % (cur_nimg))
                misc.save_pkl((E, G, D, Gs), pkl_drive)

    misc.save_pkl((E, G, D, Gs), os.path.join(submit_config.run_dir, 'network-final.pkl'))
    summary_log.close()
