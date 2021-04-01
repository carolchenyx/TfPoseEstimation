
import tensorflow as tf
import tensorflow.contrib.slim as slim
from opt import opt
from tensorflow.python.framework import ops

class LayerProvider:

    def __init__(self, is4Train):

        self.init_xavier = tf.contrib.layers.xavier_initializer()
        self.init_norm = tf.truncated_normal_initializer(stddev=0.01)
        self.init_zero = slim.init_ops.zeros_initializer()
        self.l2_regularizer = tf.contrib.layers.l2_regularizer(0.00003)

        self.is4Train = is4Train

    def inverted_bottleneck(self, inputs, up_channel_rate, channels, stride , k_s=3, dilation=1.0, scope=""):

        with tf.variable_scope("inverted_bottleneck_%s" % scope):
            with slim.arg_scope([slim.batch_norm],
                                decay=0.999,
                                fused=True,
                                is_training=self.is4Train):
                #stride = 2 if subsample else 1

                output = slim.convolution2d(inputs,
                                            up_channel_rate * inputs.get_shape().as_list()[-1],
                                            stride=1,
                                            kernel_size=[1, 1],
                                            weights_initializer=self.init_xavier,
                                            biases_initializer=self.init_zero,
                                            activation_fn=tf.nn.swish,
                                            normalizer_fn=slim.batch_norm,
                                            weights_regularizer=None,
                                            scope=scope + '_up_pointwise',
                                            trainable=self.is4Train)

                output = slim.separable_convolution2d(output,
                                                      num_outputs=None,
                                                      stride=stride,
                                                      depth_multiplier=opt.depth_multiplier,
                                                      activation_fn=tf.nn.swish,
                                                      kernel_size=k_s,
                                                      weights_initializer=self.init_xavier,
                                                      weights_regularizer=self.l2_regularizer,
                                                      biases_initializer=None,
                                                      normalizer_fn=slim.batch_norm,
                                                      rate=dilation,
                                                      padding="SAME",
                                                      scope=scope + '_depthwise',
                                                      trainable=self.is4Train)

                output = slim.convolution2d(output,
                                            channels,
                                            stride=1,
                                            kernel_size=[1, 1],
                                            activation_fn=None,
                                            weights_initializer=self.init_xavier,
                                            biases_initializer=self.init_zero,
                                            normalizer_fn=slim.batch_norm,
                                            weights_regularizer=None,
                                            scope=scope + '_pointwise',
                                            trainable=self.is4Train)

                if inputs.get_shape().as_list()[1:] == output.get_shape().as_list()[1:]:
                    output = tf.add(inputs, output)
                print("")
        return output

    def convb(self, input, k_h, k_w, c_o, stride, name, relu=True):

        with slim.arg_scope([slim.batch_norm], decay=0.999, fused=True, is_training=self.is4Train):
            output = slim.convolution2d(
                inputs=input,
                num_outputs=c_o,
                kernel_size=[k_h, k_w],
                stride=stride,
                normalizer_fn=slim.batch_norm,
                weights_regularizer=self.l2_regularizer,
                weights_initializer=self.init_xavier,
                biases_initializer=self.init_zero,
                activation_fn=tf.nn.relu if relu else None,
                scope=name,
                trainable=self.is4Train)

        return output

    # def tf_conv2d_transpose(self,input,weight):
    #     input_shape = input.get_shape().as_list()
    #     output_shape = [input_shape[0], input_shape[1]*2 , input_shape[2]*2 , weights_shape[2]]
    #
    #     with ops.name_scope(name, "conv2d_transpose",[value, filter, output_shape]) as name:
    #         return tf.nn.conv2d_transpose(value,filter,output_shape,strides,padding=padding,data_format=data_format,dilations=dilations,name=name)
    #
    #     return output

    def pointwise_convolution(self, inputs, channels, scope=""):
        with tf.variable_scope("merge_%s" % scope):
            with slim.arg_scope([slim.batch_norm],
                                decay=0.999,
                                fused=True,
                                is_training=self.is4Train):
                return slim.convolution2d(inputs,
                                          channels,
                                          stride=1,
                                          kernel_size=[1, 1],
                                          activation_fn=None,
                                          weights_initializer=self.init_xavier,
                                          biases_initializer=self.init_zero,
                                          normalizer_fn=slim.batch_norm,
                                          weights_regularizer=None,
                                          scope=scope + '_pointwise',
                                          trainable=self.is4Train
                                          )

    def relu6(x, name='relu6'):
        return tf.nn.relu6(x, name)


    def swish(x, name='swish'):
        return tf.nn.swish(x,name)
        # x*tf.nn.sigmoid(x, name)

    def leakyrelu(x, leak=0.1, name='leakyrelu'):
        with tf.variable_scope(name):
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            f3 = f1 * x + f2 * tf.abs(x)
            return f3

def _fully_connected_layer(inputs, units, name="fc", activation=None, use_bias=True, reuse=None):
    return tf.layers.dense(inputs, units, activation=activation, use_bias=use_bias,
                           name=name, reuse=reuse)


def _global_avg(inputs, pool_size, strides, padding='valid', name='global_avg'):
    return tf.layers.average_pooling2d(inputs, pool_size, strides,
                                       padding=padding, data_format='channels_last', name=name)


