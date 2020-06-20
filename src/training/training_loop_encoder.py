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
        if False: #mirror_augment: #? deactivate for now
            with tf.name_scope('MirrorAugment'):
                s = tf.shape(x)
                mask = tf.random_uniform([s[0], 1, 1, 1], 0.0, 1.0)
                mask = tf.tile(mask, [1, s[1], s[2], s[3]])
                x = tf.where(mask < 0.5, x, tf.reverse(x, axis=[3]))
        return x

#
# defines how a single example in is stored in the tfrecord
# returns: single sample
# in our case with updated dataset: output of shape [2, ..., ..., ...] -> stack
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


#
# returns a batch of data
# -> batch of stacks
def get_train_data(sess, data_dir, submit_config, mode):
    if mode == 'train':
        shuffle = False; repeat = True; batch_size = submit_config.batch_size
    elif mode == 'test':
        shuffle = False; repeat = True; batch_size = submit_config.batch_size_test
    elif mode == 'train_secondary':
        shuffle = True; repeat = True; batch_size = submit_config.batch_size
    elif mode == 'test_secondary':
        shuffle = True; repeat = True; batch_size = submit_config.batch_size_test
    else:
        raise Exception("mode must in ['train', 'test', 'train_secondary', 'test_secondary'], but got {}" % mode)

    if (mode == 'train') | (mode == 'train_secondary'):
        #get list of all tfrecord files in the dataset folder
        records_path_list = os.listdir(path=data_dir)
        for i in range(len(records_path_list)):
            records_path_list[i] = os.path.join(data_dir, records_path_list[i])
    
        dset = tf.data.TFRecordDataset(records_path_list)
    else:
        dset = tf.data.TFRecordDataset(data_dir)
        
    dset = dset.map(parse_tfrecord_tf, num_parallel_calls=16) # we can still take the whole [2, ..., ..., ...] sample here
    if shuffle:
        bytes_per_item = np.prod([2, 3, submit_config.image_size, submit_config.image_size]) * np.dtype('uint8').itemsize # 2x byte size!!!
        dset = dset.shuffle(((4096 << 20) - 1) // bytes_per_item + 1) #?
    if repeat:
        dset = dset.repeat()
    dset = dset.batch(batch_size)
    train_iterator = tf.data.Iterator.from_structure(dset.output_types, dset.output_shapes)
    training_init_op = train_iterator.make_initializer(dset)
    stack_batch = train_iterator.get_next()
    sess.run(training_init_op)
    return stack_batch


def test(E, Gs, real_portraits_test, real_landmarks_test, real_shuffled_test, submit_config):
    with tf.name_scope("Run"), tf.control_dependencies(None):
        with tf.device("/cpu:0"):
            in_split_landmarks = tf.split(real_landmarks_test, num_or_size_splits=submit_config.num_gpus, axis=0)
            in_split_portraits = tf.split(real_portraits_test, num_or_size_splits=submit_config.num_gpus, axis=0)
            in_split_shuffled = tf.split(real_shuffled_test, num_or_size_splits=submit_config.num_gpus, axis=0)
        out_split = []
        num_layers, latent_dim = Gs.components.synthesis.input_shape[1:3]
        for gpu in range(submit_config.num_gpus):
            with tf.device("/gpu:%d" % gpu):
                in_landmarks_gpu = in_split_landmarks[gpu]
                in_portraits_gpu = in_split_portraits[gpu]
                in_shuffled_gpu = in_split_shuffled[gpu]
                
                #create images for non matching portraits and landmarks
                latent_w = E.get_output_for(in_shuffled_gpu, in_landmarks_gpu, phase=False)
                latent_wp = tf.reshape(latent_w, [in_shuffled_gpu.shape[0], num_layers, latent_dim])
                fake_X_val = Gs.components.synthesis.get_output_for(latent_wp, randomize_noise=False)
                
                out_split.append(fake_X_val)
                
                #create images for matching portraits and landmarks
                latent_w = E.get_output_for(in_portraits_gpu, in_landmarks_gpu, phase=False)
                latent_wp = tf.reshape(latent_w, [in_portraits_gpu.shape[0], num_layers, latent_dim])
                fake_X_val = Gs.components.synthesis.get_output_for(latent_wp, randomize_noise=False)
                
                out_split.append(fake_X_val)

        with tf.device("/cpu:0"):
            out_expr = tf.concat(out_split, axis=0)

    return out_expr

def sample_random_portraits(Gs, batch_size):
    random_latent_z = tf.random.normal(shape=[batch_size, 512])
    random_portraits = Gs.get_output_for(random_latent_z, np.zeros((batch_size, 0)), is_training=False)
    return random_portraits

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
        placeholder_real_portraits_train = tf.placeholder(tf.float32, [submit_config.batch_size, 3, submit_config.image_size, submit_config.image_size], name='placeholder_real_portraits_train')
        placeholder_real_landmarks_train = tf.placeholder(tf.float32, [submit_config.batch_size, 3, submit_config.image_size, submit_config.image_size], name='placeholder_real_landmarks_train')
        placeholder_real_shuffled_train = tf.placeholder(tf.float32, [submit_config.batch_size, 3, submit_config.image_size, submit_config.image_size], name='placeholder_real_shuffled_train')

        placeholder_real_portraits_test = tf.placeholder(tf.float32, [submit_config.batch_size_test, 3, submit_config.image_size, submit_config.image_size], name='placeholder_real_portraits_test')
        placeholder_real_landmarks_test = tf.placeholder(tf.float32, [submit_config.batch_size_test, 3, submit_config.image_size, submit_config.image_size], name='placeholder_real_landmarks_test')
        placeholder_real_shuffled_test = tf.placeholder(tf.float32, [submit_config.batch_size_test, 3, submit_config.image_size, submit_config.image_size], name='placeholder_real_shuffled_test')

        real_split_landmarks = tf.split(placeholder_real_landmarks_train, num_or_size_splits=submit_config.num_gpus, axis=0)
        real_split_portraits = tf.split(placeholder_real_portraits_train, num_or_size_splits=submit_config.num_gpus, axis=0)
        real_split_shuffled = tf.split(placeholder_real_shuffled_train, num_or_size_splits=submit_config.num_gpus, axis=0)
        
        placeholder_training_flag = tf.placeholder(tf.string, name='placeholder_training_flag')

    with tf.device('/gpu:0'):
        if resume_run_id is not None:
            network_pkl = misc.locate_network_pkl(resume_run_id, resume_snapshot)
            print('Loading networks from "%s"...' % network_pkl)
            _E, _G, _D, _Gs = misc.load_pkl(network_pkl)
            start = int(network_pkl.split('-')[-1].split('.')[0]) // submit_config.batch_size
            print('Start: ', start)
        else:
            print('Constructing networks...')
            G, _, Gs = misc.load_pkl(decoder_pkl.decoder_pkl) # don't use pre-trained discriminator!
            num_layers = Gs.components.synthesis.input_shape[1]

            # here we add a new discriminator!
            D = tflib.Network('D',  # name of the network how we call it
                              num_channels=3, resolution=128, label_size=0,  #some needed for this build function
                              func_name="training.networks_stylegan.D_basic") # function of that network. more was not passed in d_args!
                              # input is not passed here (just construction - note that we do not call the actual function!). Instead, network will inspect build function and require it for the get_output_for function.
            print("Created new Discriminator!")

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
            real_portraits_gpu = process_reals(real_split_portraits[gpu], mirror_augment, drange_data, drange_net)
            shuffled_portraits_gpu = process_reals(real_split_shuffled[gpu], mirror_augment, drange_data, drange_net)
            real_landmarks_gpu = process_reals(real_split_landmarks[gpu], mirror_augment, drange_data, drange_net)
            with tf.name_scope('E_loss'), tf.control_dependencies(None):
                E_loss, recon_loss, adv_loss = dnnlib.util.call_func_by_name(E=E_gpu, G=G_gpu, D=D_gpu, perceptual_model=perceptual_model, real_portraits=real_portraits_gpu, shuffled_portraits=shuffled_portraits_gpu, real_landmarks=real_landmarks_gpu, training_flag=placeholder_training_flag, **E_loss_args) # change signature in loss
                E_loss_rec += recon_loss
                E_loss_adv += adv_loss
            with tf.name_scope('D_loss'), tf.control_dependencies(None):
                D_loss, loss_fake, loss_real, loss_gp = dnnlib.util.call_func_by_name(E=E_gpu, G=G_gpu, D=D_gpu, real_portraits=real_portraits_gpu, shuffled_portraits=shuffled_portraits_gpu, real_landmarks=real_landmarks_gpu, training_flag=placeholder_training_flag, **D_loss_args) # change signature in ...
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
    fake_X_val = test(E, Gs, placeholder_real_portraits_test, placeholder_real_landmarks_test, placeholder_real_shuffled_test, submit_config)
    
    #sampled_portraits_val = sample_random_portraits(Gs, submit_config.batch_size)
    #sampled_portraits_val_test = sample_random_portraits(Gs, submit_config.batch_size_test)

    sess = tf.get_default_session()

    print('Getting training data...')
    # x_batch is a batch of (2, ..., ..., ...) records!
    stack_batch_train = get_train_data(sess, data_dir=dataset_args.data_train, submit_config=submit_config, mode='train')
    stack_batch_test = get_train_data(sess, data_dir=dataset_args.data_test, submit_config=submit_config, mode='test')
    
    stack_batch_train_secondary = get_train_data(sess, data_dir=dataset_args.data_train, submit_config=submit_config, mode='train_secondary')
    stack_batch_test_secondary = get_train_data(sess, data_dir=dataset_args.data_test, submit_config=submit_config, mode='test_secondary')

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
    
    
    # here is the actual training loop: all iterations
    for it in range(start, max_iters):
        batch_stacks = sess.run(stack_batch_train)
        batch_portraits = batch_stacks[:,0,:,:,:]
        batch_landmarks = batch_stacks[:,1,:,:,:]
        
        batch_stacks_secondary = sess.run(stack_batch_train_secondary)
        batch_shuffled = batch_stacks_secondary[:,0,:,:,:]
        
        training_flag = "appearance" if it % 2 == 0 else "pose"
        
        feed_dict_1 = {placeholder_real_portraits_train: batch_portraits, placeholder_real_landmarks_train: batch_landmarks, placeholder_real_shuffled_train: batch_shuffled, placeholder_training_flag: training_flag}

        # here we query these encoder- and discriminator losses. as input we provide: batch_stacks = batch of images + landmarks.
        _, recon_, adv_ = sess.run([E_train_op, E_loss_rec, E_loss_adv], feed_dict_1)
        _, d_r_, d_f_, d_g_= sess.run([D_train_op, D_loss_real, D_loss_fake, D_loss_grad], feed_dict_1)

        cur_nimg += submit_config.batch_size

        if it % 50 == 0:
            print('Iter: %06d recon_loss: %-6.4f adv_loss: %-6.4f d_r_loss: %-6.4f d_f_loss: %-6.4f d_reg: %-6.4f time:%-12s' % (
                it, recon_, adv_, d_r_, d_f_, d_g_, dnnlib.util.format_time(time.time() - start_time)))
            sys.stdout.flush()
            tflib.autosummary.save_summaries(summary_log, it)
            
            
            
            
        if it % 500 == 0:
            batch_stacks_test = sess.run(stack_batch_test)
            batch_portraits_test = batch_stacks_test[:,0,:,:,:]
            batch_landmarks_test = batch_stacks_test[:,1,:,:,:]
            
            batch_stacks_test_secondary = sess.run(stack_batch_test_secondary)
            batch_shuffled_test = batch_stacks_test_secondary[:,0,:,:,:]
            
            
            batch_shuffled_test_vis = misc.adjust_dynamic_range(batch_shuffled_test.astype(np.float32), [0, 255], [-1., 1.])
            batch_landmarks_test_vis = misc.adjust_dynamic_range(batch_landmarks_test.astype(np.float32), [0, 255], [-1., 1.])
            batch_portraits_test_vis = misc.adjust_dynamic_range(batch_portraits_test.astype(np.float32), [0, 255], [-1., 1.])

            batch_landmarks_test = misc.adjust_dynamic_range(batch_landmarks_test.astype(np.float32), [0, 255], [-1., 1.])
            batch_portraits_test = misc.adjust_dynamic_range(batch_portraits_test.astype(np.float32), [0, 255], [-1., 1.])
            batch_shuffled_test = misc.adjust_dynamic_range(batch_shuffled_test.astype(np.float32), [0, 255], [-1., 1.])


            debug_samples = sess.run(fake_X_val, feed_dict={placeholder_real_portraits_test: batch_portraits_test, placeholder_real_landmarks_test: batch_landmarks_test, placeholder_real_shuffled_test:batch_shuffled_test})

            debug_portraits = np.concatenate([batch_shuffled_test_vis, batch_portraits_test_vis], axis=0)
            debug_landmarks = np.concatenate([batch_landmarks_test_vis, batch_landmarks_test_vis], axis=0)
            debug_img = np.concatenate([debug_portraits, debug_landmarks, debug_samples], axis=0)
            debug_img = adjust_pixel_range(debug_img)
            debug_img = fuse_images(debug_img, row=3, col=submit_config.batch_size_test*2)
            # save image results during training, first row is original images and the second row is reconstructed images
            save_image('%s/iter_%08d.png' % (submit_config.run_dir, cur_nimg), debug_img)

            # save image to gdrive
            img_path = os.path.join(config.getGdrivePath(), 'images', ('iter_%08d.png' % (cur_nimg)))
            save_image(img_path, debug_img)

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