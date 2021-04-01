import tensorflow as tf
from opt import opt
import numpy as np
import math
from config import config_cmd as config

def buildLoss(heatmapGT, outputStages, lossName):
    batchSize = opt.batch
    jointweigt = config.joint_weight
    usejointweight_ornot = config.use_different_joint_weights
    losses = []
    lossesALL = []
    for idx, stage_out in enumerate(outputStages):
        loss, lossess = posenetLoss(heatmapGT, stage_out, lossName + '_' + str(idx), batchSize, jointweigt,
                                 usejointweight_ornot)
        tf.summary.scalar(lossName + "_stage_" + str(idx), (tf.reduce_sum(loss) / batchSize))
        losses.append(loss)
        lossesALL.append(lossess)

    return lossesALL, (tf.reduce_sum(losses) / len(outputStages)) / batchSize


def posenetLoss_nooffset(gt, pred, lossName, batchSize):
    predHeat, gtHeat = pred[:, :, :, :opt.totaljoints], gt[:, :, :, :opt.totaljoints]
    totaljoints = opt.totaljoints

    if opt.hm_lossselect == 'l2':
        heatmapLoss = tf.nn.l2_loss(predHeat - gtHeat, name=lossName + "_heatmapLoss")
    elif opt.hm_lossselect == 'wing':
        heatmapLoss = wing_loss(predHeat, gtHeat)
    elif opt.hm_lossselect == 'adaptivewing':
        heatmapLoss = adaptivewingLoss(predHeat, gtHeat)
    elif opt.hm_lossselect == 'smooth_l1':
        heatmapLoss = smooth_l1_loss(None, predHeat, gtHeat)
    else:
        raise ValueError("Your optimizer name is wrong")

    for recordId in range(batchSize):
        for jointId in range(totaljoints):
            print(str(recordId) + "/" + str(batchSize) + " : " + str(jointId))
            # ================================> decode <x,y> from gt heatmap
            inlinedPix = tf.reshape(gtHeat[recordId, :, :, jointId], [-1])
            pixId = tf.argmax(inlinedPix)
            x = tf.floormod(pixId, gtHeat.shape[2])
            y = tf.cast(tf.divide(pixId, gtHeat.shape[2]), tf.int64)

    print("huber loss built")
    tf.summary.scalar(lossName + "_heatmapLoss", heatmapLoss)
    return heatmapLoss

def posenetLoss(gt, pred, lossName, batchSize, jointweigt, usejointweight_ornot):
    predHeat, gtHeat = pred[:, :, :, :opt.totaljoints], gt[:, :, :, :opt.totaljoints]

    if usejointweight_ornot == True:
        target_weight = np.ones((1, 56, 56, opt.totaljoints),
                                dtype=np.float32)
        target_weight = np.multiply(target_weight, jointweigt)
        predHeat = tf.multiply(predHeat, target_weight)
        gtHeat = tf.multiply(gtHeat, target_weight)

    predOffX, gtOffX = pred[:, :, :, opt.totaljoints:(2 * opt.totaljoints)], gt[:, :, :, opt.totaljoints:(
            2 * opt.totaljoints)]
    predOffY, gtOffY = pred[:, :, :, (2 * opt.totaljoints):], gt[:, :, :, (2 * opt.totaljoints):]
    totaljoints = opt.totaljoints

    if opt.hm_lossselect == 'l2':
        heatmapLoss = tf.nn.l2_loss(predHeat - gtHeat, name=lossName + "_heatmapLoss")
    elif opt.hm_lossselect == 'wing':
        heatmapLoss = wing_loss(predHeat, gtHeat)
    elif opt.hm_lossselect == 'adaptivewing':
        heatmapLoss = adaptivewingLoss(predHeat, gtHeat)
    elif opt.hm_lossselect == 'smooth_l1':
        heatmapLoss = smooth_l1_loss(None, predHeat, gtHeat)
    else:
        raise ValueError("Your optimizer name is wrong")
    offsetGT, offsetPred = [], []

    for recordId in range(batchSize):
        for jointId in range(totaljoints):
            print(str(recordId) + "/" + str(batchSize) + " : " + str(jointId))
            # ================================> decode <x,y> from gt heatmap

            inlinedPix = tf.reshape(gtHeat[recordId, :, :, jointId], [-1])
            pixId = tf.argmax(inlinedPix)

            x = tf.floormod(pixId, gtHeat.shape[2])
            y = tf.cast(tf.divide(pixId, gtHeat.shape[2]), tf.int64)

            # ==============================> add offset loss over the gt pix

            offsetGT.append(gtOffX[recordId, y, x, jointId])
            offsetPred.append(predOffX[recordId, y, x, jointId])
            offsetGT.append(gtOffY[recordId, y, x, jointId])
            offsetPred.append(predOffY[recordId, y, x, jointId])

    print("start building huber loss")
    offsetGT = tf.stack(offsetGT, 0)
    offsetPred = tf.stack(offsetPred, 0)
    offsetLoss = 5 * tf.losses.huber_loss(offsetGT, offsetPred)
    print("huber loss built")

    tf.summary.scalar(lossName + "_heatmapLoss", heatmapLoss)
    tf.summary.scalar(lossName + "_offsetLoss", offsetLoss)
    ht = predHeat - gtHeat

    return (heatmapLoss + offsetLoss), ht


def wing_loss(predHeat, gtHeat, w=10.0, epsilon=2.0):
    with tf.name_scope('wing_loss'):
        x = predHeat - gtHeat
        c = w * (1.0 - math.log(1.0 + w / epsilon))
        absolute_x = tf.abs(x)
        losses = tf.where(
            tf.greater(w, absolute_x),
            w * tf.log(1.0 + absolute_x / epsilon),
            absolute_x - c
        )
        loss = tf.reduce_mean(tf.reduce_sum(losses))
        tf.cast(loss, tf.float32)
        return loss



def adaptivewingLoss(predHeat, gtHeat, omega=14.0, theta=0.5, epsilon=1.0, alpha=2.1):
    delta_y = predHeat-gtHeat
    delta_y = tf.abs(delta_y)
    delta_y1 = delta_y[delta_y < theta]
    a = tf.div(delta_y1,omega)
    b = tf.div(epsilon,theta)
    delta_y2 = delta_y[delta_y >= theta]
    y1 = delta_y[delta_y < theta]
    y2 = delta_y[delta_y >= theta]
    # y[delta_y >= theta]
    loss1 = omega * tf.log(1 + tf.pow(a, alpha - y1))
    A = omega * (1 / (1 + tf.pow(b, alpha - y2))) * (alpha - y2) * (tf.pow(b, alpha - y2 - 1)) * (1 / epsilon)
    C = theta * A - omega * tf.log(1 + tf.pow(b, alpha - y2))
    loss2 = A * delta_y2 - C

    data_len = tf.size(loss1) + tf.size(loss2)
    data_len = tf.cast(data_len,tf.float32)
    return tf.div((tf.reduce_sum(loss1)+tf.reduce_sum(loss2)),data_len)
    # return (loss1.sum() + loss2.sum()) /(len(loss1) + len(loss2))

def smooth_l1_loss(self, predHeat, gtHeat, scope=tf.GraphKeys.LOSSES):
    with tf.variable_scope(scope):
        absolute_x = tf.abs(predHeat - gtHeat)
        sq_x = 0.5*absolute_x**2
        less_one = tf.cast(tf.less(absolute_x, 1.0), tf.float32)
        smooth_l1_loss = (less_one*sq_x)+(1.0-less_one)
        return smooth_l1_loss