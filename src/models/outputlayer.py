from src.models.Layerprovider1 import LayerProvider as LayerProvider1
from src.models.Layerprovider import LayerProvider
import tensorflow as tf
from opt import opt
from config import config_cmd as config
import tensorflow.contrib.slim as slim

class finallayerforoffsetoption(object):
    def __init__(self, offset = opt.offset,pixel = config.pixelshuffle,conv = config.convb_13):
        self.lProvider1 = LayerProvider1(opt.isTrainpre)
        self.lProvider = LayerProvider(opt.isTrainpre)
        self.offset = offset
        self.pixel = pixel
        self.conv = conv

    def fornetworks_noDUC(self,output, totalJoints):

        output = tf.nn.conv2d_transpose(output, filter=tf.Variable(tf.random.normal(shape=[4, 4, 1280, 320])), output_shape=[1,28,28,320],
                                         strides=[1, 3, 3, 1], padding='SAME')
        for i in range(len(self.pixel)):
            output = self.lProvider1.convb(output, config.convb_16[i][0], config.convb_16[i][1],
                                           config.convb_16[i][2], config.convb_16[i][3],
                                           config.convb_16[i][4], relu=True)
        output = tf.nn.conv2d_transpose(output , filter=tf.Variable(tf.random.normal(shape=[4, 4, 208, 52])), output_shape=[1,112,112,52],strides=[1,3, 3, 1], padding='SAME')

        if self.offset == True:
            seg = self.lProvider1.pointwise_convolution(output, totalJoints, scope= "output-1")

            seg = tf.sigmoid(seg)

            reg = self.lProvider1.pointwise_convolution(output, totalJoints * 2, scope="output-2")

            self.output = tf.concat([seg, reg], 3, name="Output")

        else:
            output = self.lProvider1.pointwise_convolution(output, totalJoints, scope="Output")

            self.output = tf.identity(output, "Output")

        return self.output

    def fornetworks_DUC(self, output,totalJoints,layers):
        if opt.Shuffle == 2:
            if opt.backbone == "resnet18":
                output = tf.nn.depth_to_space(output, 2)  # pixel shuffle,name = 'depth_to_space_in_2'

            else:
                output = tf.nn.depth_to_space(output, 2) # pixel shuffle,name = 'depth_to_space_in_2'

            for i in range(len(self.pixel)):
                if opt.totaljoints == 13:
                    output = self.lProvider1.convb(output, self.conv[i][0], self.conv[i][1], self.conv[i][2], self.conv[i][3],
                                          self.conv[i][4], relu=True)

                elif opt.totaljoints == 16:
                    output = self.lProvider1.convb(output, config.convb_16[i][0], config.convb_16[i][1], config.convb_16[i][2], config.convb_16[i][3],
                                                  config.convb_16[i][4], relu=True)
                output = tf.nn.depth_to_space(output, self.pixel[i])#name = 'depth_to_space_out_2'

        elif opt.Shuffle == 4:
            if opt.backbone == "resnet18":
                output = tf.nn.depth_to_space(output, 2)  # pixel shuffle,name = 'depth_to_space_in_2'

            else:
                output = tf.nn.depth_to_space(output, 2) # pixel shuffle,name = 'depth_to_space_in_2'

            for i in range(len(self.pixel)):
                output = self.lProvider1.convb(output, config.convb_16[i][0], config.convb_16[i][1],
                                               config.convb_16[i][2], config.convb_16[i][3],
                                               config.convb_16[i][4], relu=True)
            output = tf.nn.depth_to_space(output, 4, name='depth_to_space_4')

        if self.offset == False:
            output = tf.identity(output, "Output")

            # return a tensor with the same shape and contents as input.
        else:
            seg = self.lProvider1.pointwise_convolution(output, totalJoints, scope="output-1")

            seg = tf.sigmoid(seg)

            reg = self.lProvider1.pointwise_convolution(output, totalJoints * 2, scope="output-2")

            output = tf.concat([seg, reg], 3, name="Output")

            # output = tf.identity(output, "Output")
        return output
    def _get_deconv_cfg(deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding

