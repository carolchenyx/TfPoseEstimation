from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow.compat.v1 as tf
from config.config import Config
from opt import opt
from src.models.posenetv2 import PoseNetv2
from src.utils.loss import buildLoss


def model_fn(features, labels, mode,params,config=None):
    # features,  # This is batch_features from input_fn,`Tensor` or dict of `Tensor` (depends on data passed to `fit`).
    # labels,  # This is batch_labels from input_fn
    # mode,  # An instance of tf.estimator.ModeKeys
    # params,  # Additional configuration
    model = PoseNetv2((None,224,224,3),is4Train=True)
    output = model.output
    learningRate = tf.placeholder(tf.float32, [], name='learningRate')
    globalStep = tf.Variable(0, trainable=False)
    updater = []
    if opt.lr_type == "exponential_decay":
        lr = tf.train.exponential_decay(learningRate, global_step=globalStep,
                                             decay_steps=opt.decay_steps, decay_rate=opt.decay_rate, staircase=True)
    elif opt.lr_type == "cosine_decay":
        lr = tf.train.cosine_decay(learningRate, global_step=globalStep,
                                        decay_steps=opt.decay_steps, alpha=0.0, name=None)
    elif opt.lr_type == "inverse_time_decay":
        lr = tf.train.inverse_time_decay(learningRate, global_step=globalStep,
                                              decay_steps=opt.decay_steps, decay_rate=opt.decay_rate,
                                              staircase=False, name=None)
    elif opt.lr_type == "polynomial_decay":
        lr = tf.train.polynomial_decay(learningRate, global_step=globalStep,
                                            decay_steps=opt.decay_steps,
                                            power=1.0, cycle=False, name=None)
    else:
        raise ValueError("Your lr_type name is wrong")
    if opt.optimizer == "Adam":
        optimize = tf.train.AdamOptimizer(lr, epsilon=opt.epsilon)
    elif opt.optimizer == "Momentum":  # use_locking: 为True时锁定更新
        optimize = tf.train.MomentumOptimizer(lr, momentum=opt.momentum, use_locking=False, name='Momentum',
                                              use_nesterov=False)
    elif opt.optimizer == "Gradient":
        optimize = tf.train.GradientDescentOptimizer(lr,
                                                     use_locking=False, name='GrandientDescent')
    else:
        raise ValueError("Your optimizer name is wrong")
    if opt.offset == True:
        heatmapGT = tf.placeholder(tf.float32,
                                        shape=(None, output.shape[1], output.shape[2], opt.totaljoints * 3),
                                        name='heatmapGT')
    else:
        heatmapGT = tf.placeholder(tf.float32, shape=(None, output.shape[1], output.shape[2], opt.totaljoints),
                                        name='heatmapGT')
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    lossall, Loss = buildLoss(heatmapGT, [output],"trainLoss")

    grads = optimize.compute_gradients(Loss)
    # self.apply_gradient_op = self.opt.apply_gradients(self.grads, global_step=self.globalStep)

    with tf.control_dependencies(update_ops):
        # self.train_op = tf.group(apply_gradient_op, variables_averages_op)
        train_op = optimize.minimize(Loss,globalStep)
    updater.append(train_op)



def main(m_dir=None):

  cfg = Config.fromfile(opt.config)
  params = dict(
      steps_per_epoch=opt.total_traindata / opt.batch,
      use_bfloat16=opt.use_bfloat16)
  run_cfg = tf.estimator.RunConfig(
      model_dir=m_dir,
      tf_random_seed=2,
      save_summary_steps=2,
      save_checkpoints_steps=10,
      keep_checkpoint_max=1)

  model_fn(0,0,0,0)
  estimator = tf.estimator.Estimator(model_fn, model_dir=None, config=run_cfg, params=None,
               warm_start_from=None)



if __name__ == '__main__':
    # run.run(main)
    main()
