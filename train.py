from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow.compat.v1 as tf
from config.config import Config
from opt import opt
import time


def model_fn(features, labels, mode, params):
  """The model_fn to be used with TPUEstimator.

  Args:
    features: `Tensor` of batched images.
    labels: `Tensor` of one hot labels for the data samples
    mode: one of `tf.estimator.ModeKeys.{TRAIN,EVAL,PREDICT}`
    params: `dict` of parameters passed to the model from the TPUEstimator,
        `params['batch_size']` is always provided and should be used as the
        effective batch size.

  Returns:
    A `TPUEstimatorSpec` for the model
  """
  if isinstance(features, dict):
    features = features['feature']

  # In most cases, the default data format NCHW instead of NHWC should be
  # used for a significant performance boost on GPU. NHWC should be used
  # only if the network needs to be run on CPU since the pooling operations
  # are only supported on NHWC. TPU uses XLA compiler to figure out best layout.
  if FLAGS.data_format == 'channels_first':
    assert not FLAGS.transpose_input    # channels_first only for GPU
    features = tf.transpose(features, [0, 3, 1, 2])
    stats_shape = [3, 1, 1]
  else:
    stats_shape = [1, 1, 3]

  if FLAGS.transpose_input and mode != tf.estimator.ModeKeys.PREDICT:
    features = tf.transpose(features, [3, 0, 1, 2])  # HWCN to NHWC

  is_training = (mode == tf.estimator.ModeKeys.TRAIN)
  has_moving_average_decay = (FLAGS.moving_average_decay > 0)
  # This is essential, if using a keras-derived model.
  tf.keras.backend.set_learning_phase(is_training)
  logging.info('Using open-source implementation.')
  override_params = {}
  if FLAGS.batch_norm_momentum is not None:
    override_params['batch_norm_momentum'] = FLAGS.batch_norm_momentum
  if FLAGS.batch_norm_epsilon is not None:
    override_params['batch_norm_epsilon'] = FLAGS.batch_norm_epsilon
  if FLAGS.dropout_rate is not None:
    override_params['dropout_rate'] = FLAGS.dropout_rate
  if FLAGS.survival_prob is not None:
    override_params['survival_prob'] = FLAGS.survival_prob
  if FLAGS.data_format:
    override_params['data_format'] = FLAGS.data_format
  if FLAGS.num_label_classes:
    override_params['num_classes'] = FLAGS.num_label_classes
  if FLAGS.depth_coefficient:
    override_params['depth_coefficient'] = FLAGS.depth_coefficient
  if FLAGS.width_coefficient:
    override_params['width_coefficient'] = FLAGS.width_coefficient

  def normalize_features(features, mean_rgb, stddev_rgb):
    """Normalize the image given the means and stddevs."""
    features -= tf.constant(mean_rgb, shape=stats_shape, dtype=features.dtype)
    features /= tf.constant(stddev_rgb, shape=stats_shape, dtype=features.dtype)
    return features

  def build_model():
    """Build model using the model_name given through the command line."""
    model_builder = model_builder_factory.get_model_builder(FLAGS.model_name)
    normalized_features = normalize_features(features, model_builder.MEAN_RGB,
                                             model_builder.STDDEV_RGB)
    logits, _ = model_builder.build_model(
        normalized_features,
        model_name=FLAGS.model_name,
        training=is_training,
        override_params=override_params,
        model_dir=FLAGS.model_dir)
    return logits

  if params['use_bfloat16']:
    with tf.tpu.bfloat16_scope():
      logits = tf.cast(build_model(), tf.float32)
  else:
    logits = build_model()

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'classify': tf.estimator.export.PredictOutput(predictions)
        })

  # If necessary, in the model_fn, use params['batch_size'] instead the batch
  # size flags (--train_batch_size or --eval_batch_size).
  batch_size = params['batch_size']   # pylint: disable=unused-variable

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits,
      onehot_labels=labels,
      label_smoothing=FLAGS.label_smoothing)

  # Add weight decay to the loss for non-batch-normalization variables.
  loss = cross_entropy + FLAGS.weight_decay * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()
       if 'batch_normalization' not in v.name])

  global_step = tf.train.get_global_step()
  if has_moving_average_decay:
    ema = tf.train.ExponentialMovingAverage(
        decay=FLAGS.moving_average_decay, num_updates=global_step)
    ema_vars = utils.get_ema_vars()

  host_call = None
  restore_vars_dict = None
  if is_training:
    # Compute the current epoch and associated learning rate from global_step.
    current_epoch = (
        tf.cast(global_step, tf.float32) / params['steps_per_epoch'])

    scaled_lr = FLAGS.base_learning_rate * (FLAGS.train_batch_size / 256.0)
    logging.info('base_learning_rate = %f', FLAGS.base_learning_rate)
    learning_rate = utils.build_learning_rate(scaled_lr, global_step,
                                              params['steps_per_epoch'])
    optimizer = utils.build_optimizer(learning_rate)
    if FLAGS.use_tpu:
      # When using TPU, wrap the optimizer with CrossShardOptimizer which
      # handles synchronization details between different TPU cores. To the
      # user, this should look like regular synchronous training.
      optimizer = tf.tpu.CrossShardOptimizer(optimizer)

    # Batch normalization requires UPDATE_OPS to be added as a dependency to
    # the train operation.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step)

    if has_moving_average_decay:
      with tf.control_dependencies([train_op]):
        train_op = ema.apply(ema_vars)

    if not FLAGS.skip_host_call:
      def host_call_fn(gs, lr, ce):
        """Training host call. Creates scalar summaries for training metrics.

        This function is executed on the CPU and should not directly reference
        any Tensors in the rest of the `model_fn`. To pass Tensors from the
        model to the `metric_fn`, provide as part of the `host_call`. See
        https://www.tensorflow.org/api_docs/python/tf/estimator/tpu/TPUEstimatorSpec
        for more information.

        Arguments should match the list of `Tensor` objects passed as the second
        element in the tuple passed to `host_call`.

        Args:
          gs: `Tensor with shape `[batch]` for the global_step
          lr: `Tensor` with shape `[batch]` for the learning_rate.
          ce: `Tensor` with shape `[batch]` for the current_epoch.

        Returns:
          List of summary ops to run on the CPU host.
        """
        gs = gs[0]
        # Host call fns are executed FLAGS.iterations_per_loop times after one
        # TPU loop is finished, setting max_queue value to the same as number of
        # iterations will make the summary writer only flush the data to storage
        # once per loop.
        with tf2.summary.create_file_writer(
            FLAGS.model_dir, max_queue=FLAGS.iterations_per_loop).as_default():
          with tf2.summary.record_if(True):
            tf2.summary.scalar('learning_rate', lr[0], step=gs)
            tf2.summary.scalar('current_epoch', ce[0], step=gs)

            return tf.summary.all_v2_summary_ops()

      # To log the loss, current learning rate, and epoch for Tensorboard, the
      # summary op needs to be run on the host CPU via host_call. host_call
      # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
      # dimension. These Tensors are implicitly concatenated to
      # [params['batch_size']].
      gs_t = tf.reshape(global_step, [1])
      lr_t = tf.reshape(learning_rate, [1])
      ce_t = tf.reshape(current_epoch, [1])

      host_call = (host_call_fn, [gs_t, lr_t, ce_t])

  else:
    train_op = None
    if has_moving_average_decay:
      # Load moving average variables for eval.
      restore_vars_dict = ema.variables_to_restore(ema_vars)

  eval_metrics = None
  if mode == tf.estimator.ModeKeys.EVAL:
    def metric_fn(labels, logits):
      """Evaluation metric function. Evaluates accuracy.

      This function is executed on the CPU and should not directly reference
      any Tensors in the rest of the `model_fn`. To pass Tensors from the model
      to the `metric_fn`, provide as part of the `eval_metrics`. See
      https://www.tensorflow.org/api_docs/python/tf/estimator/tpu/TPUEstimatorSpec
      for more information.

      Arguments should match the list of `Tensor` objects passed as the second
      element in the tuple passed to `eval_metrics`.

      Args:
        labels: `Tensor` with shape `[batch, num_classes]`.
        logits: `Tensor` with shape `[batch, num_classes]`.

      Returns:
        A dict of the metrics to return from evaluation.
      """
      labels = tf.argmax(labels, axis=1)
      predictions = tf.argmax(logits, axis=1)
      top_1_accuracy = tf.metrics.accuracy(labels, predictions)
      in_top_5 = tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32)
      top_5_accuracy = tf.metrics.mean(in_top_5)

      return {
          'top_1_accuracy': top_1_accuracy,
          'top_5_accuracy': top_5_accuracy,
      }

    eval_metrics = (metric_fn, [labels, logits])

  num_params = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
  logging.info('number of trainable parameters: %d', num_params)

  def _scaffold_fn():
    saver = tf.train.Saver(restore_vars_dict)
    return tf.train.Scaffold(saver=saver)

  if has_moving_average_decay and not is_training:
    # Only apply scaffold for eval jobs.
    scaffold_fn = _scaffold_fn
  else:
    scaffold_fn = None

  return tf.estimator.tpu.TPUEstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      host_call=host_call,
      eval_metrics=eval_metrics,
      scaffold_fn=scaffold_fn)


def _verify_non_empty_string(value, field_name):
  """Ensures that a given proposed field value is a non-empty string.

  Args:
    value:  proposed value for the field.
    field_name:  string name of the field, e.g. `project`.

  Returns:
    The given value, provided that it passed the checks.

  Raises:
    ValueError:  the value is not a string, or is a blank string.
  """
  if not isinstance(value, str):
    raise ValueError(
        'Bigtable parameter "%s" must be a string.' % field_name)
  if not value:
    raise ValueError(
        'Bigtable parameter "%s" must be non-empty.' % field_name)
  return value


def _select_tables_from_flags():
  """Construct training and evaluation Bigtable selections from flags.

  Returns:
    [training_selection, evaluation_selection]
  """
  project = _verify_non_empty_string(
      FLAGS.bigtable_project or FLAGS.gcp_project,
      'project')
  instance = _verify_non_empty_string(FLAGS.bigtable_instance, 'instance')
  table = _verify_non_empty_string(FLAGS.bigtable_table, 'table')
  train_prefix = _verify_non_empty_string(FLAGS.bigtable_train_prefix,
                                          'train_prefix')
  eval_prefix = _verify_non_empty_string(FLAGS.bigtable_eval_prefix,
                                         'eval_prefix')
  column_family = _verify_non_empty_string(FLAGS.bigtable_column_family,
                                           'column_family')
  column_qualifier = _verify_non_empty_string(FLAGS.bigtable_column_qualifier,
                                              'column_qualifier')
  return [
      imagenet_input.BigtableSelection(  # pylint: disable=g-complex-comprehension
          project=project,
          instance=instance,
          table=table,
          prefix=p,
          column_family=column_family,
          column_qualifier=column_qualifier)
      for p in (train_prefix, eval_prefix)
  ]


def export(est, export_dir, input_image_size=None):
  """Export graph to SavedModel and TensorFlow Lite.

  Args:
    est: estimator instance.
    export_dir: string, exporting directory.
    input_image_size: int, input image size.

  Raises:
    ValueError: the export directory path is not specified.
  """
  if not export_dir:
    raise ValueError('The export directory path is not specified.')

  if not input_image_size:
    input_image_size = FLAGS.input_image_size
  is_cond_conv = FLAGS.model_name.startswith('efficientnet-condconv')
  batch_size = 1 if is_cond_conv else None  # Use fixed batch size for condconv.

  logging.info('Starting to export model.')
  if (FLAGS.model_name.startswith('efficientnet-lite') or
      FLAGS.model_name.startswith('efficientnet-edgetpu')):
    # lite or edgetpu use binlinear for easier post-quantization.
    resize_method = tf.image.ResizeMethod.BILINEAR
  else:
    resize_method = None
  image_serving_input_fn = imagenet_input.build_image_serving_input_fn(
      input_image_size, batch_size=batch_size, resize_method=resize_method)
  est.export_saved_model(
      export_dir_base=export_dir,
      serving_input_receiver_fn=image_serving_input_fn)
#

def main():

  cfg = Config.fromfile(opt.config)
  # init the logger before other steps
  timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
  params = dict(
      steps_per_epoch=opt.total_traindata / opt.batch,
      use_bfloat16=opt.use_bfloat16)
  estimator = tf.estimator.Estimator(model_fn, model_dir=None, config=None, params=None,
               warm_start_from=None)

  def build_imagenet_input(is_training= opt.isTrain):
    """Generate ImageNetInput for training and eval."""
    if FLAGS.bigtable_instance:
      logging.info('Using Bigtable dataset, table %s', FLAGS.bigtable_table)
      select_train, select_eval = _select_tables_from_flags()
      return imagenet_input.ImageNetBigtableInput(
          is_training=is_training,
          use_bfloat16=FLAGS.use_bfloat16,
          transpose_input=FLAGS.transpose_input,
          selection=select_train if is_training else select_eval,
          num_label_classes=FLAGS.num_label_classes,
          include_background_label=include_background_label,
          augment_name=FLAGS.augment_name,
          mixup_alpha=FLAGS.mixup_alpha,
          randaug_num_layers=FLAGS.randaug_num_layers,
          randaug_magnitude=FLAGS.randaug_magnitude,
          resize_method=resize_method)
    else:
      if FLAGS.data_dir == FAKE_DATA_DIR:
        logging.info('Using fake dataset.')
      else:
        logging.info('Using dataset: %s', FLAGS.data_dir)

      return imagenet_input.ImageNetInput(
          is_training=is_training,
          data_dir=FLAGS.data_dir,
          transpose_input=FLAGS.transpose_input,
          cache=FLAGS.use_cache and is_training,
          image_size=input_image_size,
          num_parallel_calls=FLAGS.num_parallel_calls,
          use_bfloat16=FLAGS.use_bfloat16,
          num_label_classes=FLAGS.num_label_classes,
          include_background_label=include_background_label,
          augment_name=FLAGS.augment_name,
          mixup_alpha=FLAGS.mixup_alpha,
          randaug_num_layers=FLAGS.randaug_num_layers,
          randaug_magnitude=FLAGS.randaug_magnitude,
          resize_method=resize_method)

  imagenet_train = build_imagenet_input(is_training=True)
  imagenet_eval = build_imagenet_input(is_training=False)

  if FLAGS.mode == 'eval':
    eval_steps = FLAGS.num_eval_images // FLAGS.eval_batch_size
    # Run evaluation when there's a new checkpoint
    for ckpt in tf.train.checkpoints_iterator(
        FLAGS.model_dir, timeout=FLAGS.eval_timeout):
      logging.info('Starting to evaluate.')
      try:
        start_timestamp = time.time()  # This time will include compilation time
        eval_results = est.evaluate(
            input_fn=imagenet_eval.input_fn,
            steps=eval_steps,
            checkpoint_path=ckpt)
        elapsed_time = int(time.time() - start_timestamp)
        logging.info('Eval results: %s. Elapsed seconds: %d',
                     eval_results, elapsed_time)
        utils.archive_ckpt(eval_results, eval_results['top_1_accuracy'], ckpt)

        # Terminate eval job when final checkpoint is reached
        current_step = int(os.path.basename(ckpt).split('-')[1])
        if current_step >= FLAGS.train_steps:
          logging.info(
              'Evaluation finished after training step %d', current_step)
          break

      except tf.errors.NotFoundError:
        # Since the coordinator is on a different job than the TPU worker,
        # sometimes the TPU worker does not finish initializing until long after
        # the CPU job tells it to start evaluating. In this case, the checkpoint
        # file could have been deleted already.
        logging.info(
            'Checkpoint %s no longer exists, skipping checkpoint', ckpt)
  else:   # FLAGS.mode == 'train' or FLAGS.mode == 'train_and_eval'
    current_step = estimator._load_global_step_from_checkpoint_dir(FLAGS.model_dir)  # pylint: disable=protected-access,line-too-long

    logging.info(
        'Training for %d steps (%.2f epochs in total). Current'
        ' step %d.', FLAGS.train_steps,
        FLAGS.train_steps / params['steps_per_epoch'], current_step)

    start_timestamp = time.time()  # This time will include compilation time

    if FLAGS.mode == 'train':
      hooks = []
      if FLAGS.use_async_checkpointing:
        try:
          from tensorflow.contrib.tpu.python.tpu import async_checkpoint  # pylint: disable=g-import-not-at-top
        except ImportError as e:
          logging.exception(
              'Async checkpointing is not supported in TensorFlow 2.x')
          raise e

        hooks.append(
            async_checkpoint.AsyncCheckpointSaverHook(
                checkpoint_dir=FLAGS.model_dir,
                save_steps=max(100, FLAGS.iterations_per_loop)))
      est.train(
          input_fn=imagenet_train.input_fn,
          max_steps=FLAGS.train_steps,
          hooks=hooks)

    else:
      assert FLAGS.mode == 'train_and_eval'
      while current_step < FLAGS.train_steps:
        # Train for up to steps_per_eval number of steps.
        # At the end of training, a checkpoint will be written to --model_dir.
        next_checkpoint = min(current_step + FLAGS.steps_per_eval,
                              FLAGS.train_steps)
        est.train(input_fn=imagenet_train.input_fn, max_steps=next_checkpoint)
        current_step = next_checkpoint

        logging.info('Finished training up to step %d. Elapsed seconds %d.',
                     next_checkpoint, int(time.time() - start_timestamp))

        # Evaluate the model on the most recent model in --model_dir.
        # Since evaluation happens in batches of --eval_batch_size, some images
        # may be excluded modulo the batch size. As long as the batch size is
        # consistent, the evaluated images are also consistent.
        logging.info('Starting to evaluate.')
        eval_results = est.evaluate(
            input_fn=imagenet_eval.input_fn,
            steps=FLAGS.num_eval_images // FLAGS.eval_batch_size)
        logging.info('Eval results at step %d: %s',
                     next_checkpoint, eval_results)
        ckpt = tf.train.latest_checkpoint(FLAGS.model_dir)
        utils.archive_ckpt(eval_results, eval_results['top_1_accuracy'], ckpt)

      elapsed_time = int(time.time() - start_timestamp)
      logging.info('Finished training up to step %d. Elapsed seconds %d.',
                   FLAGS.train_steps, elapsed_time)
  if FLAGS.export_dir:
    export(est, FLAGS.export_dir, input_image_size)


if __name__ == '__main__':
    # run.run(main)
    main()
