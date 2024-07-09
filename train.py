# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import sys
import data_helpers
from correlationNet import correlationNet
from CNN import CNN
from AutoEncoderBenign import AutoEncoderBenign
from AutoEncoderAttack import AutoEncoderAttack
from collections import OrderedDict

#fs = sys.argv[1]
fs = '1'
a = 0.001
#USERNAME = ['fileName32_1.txt_add','fileName32_2.txt_add','fileName32_3.txt_add','fileName32_4.txt_add','fileName32_5.txt_add','fileName32_6.txt_add']
#USERNAME = ['fileName32_1.txt_add_evasion','fileName32_2.txt_add_evasion','fileName32_3.txt_add_evasion','fileName32_4.txt_add_evasion','fileName32_5.txt_add_evasion','fileName32_6.txt_add_evasion']
USERNAME = ['fileName26_1.txt_n1','fileName26_1.txt_n2','fileName26_1.txt_n3','fileName30_1.txt_n1',
		'fileName30_1.txt_n2','fileName30_1.txt_n3','fileName32_1.txt_n1','fileName32_1.txt_n2',
		'fileName32_1.txt_n3','fileName32_2.txt_p1','fileName32_2.txt_p3','fileName32_2.txt_p4',
		'fileName32_2.txt_p5','fileName32_2.txt_p6']
#USERNAME = ['fileName32_1.txt_n1','fileName32_1.txt_n2','fileName32_1.txt_n3','fileName30_1.txt_n1','fileName30_1.txt_n2',
#		'fileName30_1.txt_n3','fileName32_2.txt_p1','fileName32_2.txt_p2','fileName32_2.txt_p3','fileName32_2.txt_p4',
#		'fileName32_2.txt_p5']

LEARNING_RATE = 0.01
DROPOUT_KEEP_PROB = 0.4
L2_REG_LAMBDA = 0.0
NUM_CLASSES = 2
BATCH_SIZE = 50
EPOCHS = 4

IS_IMBALANCE_LOSS = True
#NUM_DIM = 543
#LENGTH = 440
#CONV_NUM = 440
FILTER_SIZES = 8
NUM_FILTERS = 5
GNET_HIDE_NUM = 300
#POSITION_RATE = 0.8
#NEGATION_RATE = 0.7
IS_RATE = True

IS_COMPLETELY_RANDOM = False
IS_TYPE_RANDOM = True
UNKNOWN_RATE = 0.0

CHECKPOINT_EVERY = EPOCHS * BATCH_SIZE
NUM_CHECKPOINTS = 5

MODEL_SAVE_PATH = "./to/model"
MODEL_NAME = "model.ckpt"
result = open("./loss.csv","w")

testfile = open('./testfile.csv','wb')
testlabel = open('./testlabel.csv','wb')
testfile.truncate()
testlabel.truncate()

data_list = []
date_list = []
label_list = []
lc_list = []
unknown_num = 0
negation_num = 0
f_ra = open('./representation_a_train.txt','w')
f_rb = open('./representation_b_train.txt','w')
for i in USERNAME:
	
	DATA_PATH = 'insider_data/dataset/'+i+'_add_onehot'
	LABEL_PATH = 'insider_data/dataset/'+i+'_add_label'

	data_i, label_i, _, _ = data_helpers.data_load(DATA_PATH, LABEL_PATH, NUM_CLASSES)
	NUM_DIM = len(data_i[0])
	x_i, lc_i = data_helpers.generating_x(np.array(data_i),np.array(label_i),FILTER_SIZES,NUM_DIM)
	lc_list = lc_list+lc_i
	data_list = data_list+x_i
	label_list = label_list+label_i


x = np.array(data_list)
y = np.array(label_list)
lc = np.array(lc_list)
#print y.tolist()
#statistic = open('./statistic.csv','a')
#statistic.write(fs+','+str(unknown_num)+','+str(negation_num))
#statistic.write('\n')
#statistic.close()

x_pair, y_pair = data_helpers.generating_pair(x,lc,FILTER_SIZES)

lossdir = OrderedDict()
with tf.Graph().as_default():
	sess = tf.Session()
	with sess.as_default():
		
		cnn = CNN(
				weigth = NUM_DIM,
				gNet_hide_num = GNET_HIDE_NUM,
				filter_size = FILTER_SIZES,
				num_classes = NUM_CLASSES,
				num_filters = NUM_FILTERS,
				is_imbalance_loss = IS_IMBALANCE_LOSS,
				learning_rate = LEARNING_RATE,
				l2_reg_lambda = L2_REG_LAMBDA,
				a = a)
		
		autoencoderbenign = AutoEncoderBenign(
				num_filters = NUM_FILTERS,
				learning_rate = LEARNING_RATE)

		autoencoderattack = AutoEncoderAttack(
				num_filters = NUM_FILTERS,
				learning_rate = LEARNING_RATE)

		saver = tf.train.Saver()
		sess.run(tf.global_variables_initializer())
		threshold_b = []
		threshold_a = []
		def train_step(x_batch, y_batch, epoch):
			
			
			'''
			feed_dict_gatep = { 
					cnn.input_x: x_batch,
					cnn.input_y: y_batch,
					cnn.batch_size: len(x_batch),
		 			cnn.dropout_keep_prob: DROPOUT_KEEP_PROB}
			_, loss_gatep, g_probp = sess.run([cnn.train_op_gatep, cnn.loss_gNetp, cnn.gNetp], feed_dict_gatep)
			
			feed_dict_gaten = {
					cnn.input_x: x_batch,
					cnn.input_y: y_batch,
					cnn.batch_size: len(x_batch),
					cnn.dropout_keep_prob: DROPOUT_KEEP_PROB}
			_, loss_gaten, g_probn = sess.run([cnn.train_op_gaten, cnn.loss_gNetn, cnn.gNetn], feed_dict_gaten)
			'''

			feed_dict_gatepu = {
					cnn.input_x: x_batch,
					cnn.input_y: y_batch,
					cnn.batch_size: len(x_batch),
					cnn.dropout_keep_prob: DROPOUT_KEEP_PROB}
			_, loss_gatepu, g_probpu = sess.run([cnn.train_op_gatepu, cnn.loss_gNetpu, cnn.g_probpu], feed_dict_gatepu)

			feed_dict_gatenu = {
					cnn.input_x: x_batch,
					cnn.input_y: y_batch,
					cnn.batch_size: len(x_batch),
					cnn.dropout_keep_prob: DROPOUT_KEEP_PROB}
			_, loss_gatenu, g_probnu = sess.run([cnn.train_op_gatenu, cnn.loss_gNetnu, cnn.g_probnu], feed_dict_gatenu)

			feed_dict_cnn = {
					cnn.input_x: x_batch,
					cnn.input_y: y_batch,
					cnn.batch_size:len(x_batch),
					cnn.dropout_keep_prob: DROPOUT_KEEP_PROB}
			_, step, loss_cnn, loss_n, representation, y = sess.run([cnn.train_op_cnn, cnn.global_step, cnn.loss_cNet, cnn.loss_mean_n, cnn.h_pn, cnn.gate_yu], feed_dict_cnn)
			
			rep_b = []
			rep_a = []
			for index, rep in enumerate(representation):
				if y[index,0] == 1:
					rep_b.append(rep[0].tolist())
				else:
					rep_a.append(rep[1].tolist())
			#print representation
			
			if rep_b:
				feed_dict_autoencoderbenign = {
						autoencoderbenign.representation_b: rep_b}
				_, autoencoderloss_b = sess.run([autoencoderbenign.train_op_autoencoder_b, autoencoderbenign.autoencoderloss_b], feed_dict_autoencoderbenign)
				if epoch == EPOCHS-1:
					threshold_b.append(autoencoderloss_b)
					for ri in rep_b:
						rbs = [str(rii) for rii in ri]
						f_rb.write(','.join(rbs))
						f_rb.write('\n')
			else:
				autoencoderloss_b = 'None'

			if rep_a:
				feed_dict_autoencoderattack = {
						autoencoderattack.representation_a: rep_a}
				_, autoencoderloss_a = sess.run([autoencoderattack.train_op_autoencoder_a, autoencoderattack.autoencoderloss_a], feed_dict_autoencoderattack)
				if epoch == EPOCHS-1:
					threshold_a.append(autoencoderloss_a)
					for ri in rep_a:
						ras = [str(rii) for rii in ri]
						f_ra.write(','.join(ras))
						f_ra.write('\n')
			else:
				autoencoderloss_a = 'None'
			
			epoch = str(epoch)
			if lossdir.has_key(epoch):
				lossdir[epoch].append(loss_n)
			else:
				lossdir[epoch] = [loss_n]
			print epoch, loss_n, autoencoderloss_b, autoencoderloss_a
		
		batches = data_helpers.batch_iter(zip(x, y),BATCH_SIZE, EPOCHS)
		
		for batch in batches:
			#print epoch[index]:q
			x_batch, y_batch = zip(*batch[1])
			x_batch = np.array(list(x_batch))
			#x_batch = x_batch.reshape([len(x_batch), LENGTH, -1])
			
			train_step(x_batch,y_batch,batch[0])
			
			current_step = tf.train.global_step(sess, cnn.global_step)
			if current_step % CHECKPOINT_EVERY == 0:
				saver.save( sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),global_step = current_step)

MODELB_SAVE_PATH = "./to/modelB"
MODELB_NAME = "modelb.ckpt"
with tf.Graph().as_default():
	sess = tf.Session()
	with sess.as_default():

		corNet = correlationNet(
				weigth = NUM_DIM,
				num_classes = NUM_CLASSES,
				learning_rate = LEARNING_RATE)

		saver = tf.train.Saver()
		sess.run(tf.global_variables_initializer())

		def cor_train_step(x_batch, y_batch, epoch):
			feed_dict_cp = {
					corNet.input_x: x_batch,
					corNet.input_y: y_batch}
			_, loss_cp = sess.run([corNet.train_op_p, corNet.loss_p], feed_dict_cp)

			print loss_cp

		batches = data_helpers.batch_iter(zip(x_pair,y_pair),BATCH_SIZE, EPOCHS)
		for batch in batches:
			x_batch, y_batch = zip(*batch[1])
			x_batch = np.array(list(x_batch))
			cor_train_step(x_batch,y_batch,batch[0])

			current_step = tf.train.global_step(sess, corNet.global_step)
			if current_step % CHECKPOINT_EVERY == 0:
				saver.save(sess, os.path.join(MODELB_SAVE_PATH, MODELB_NAME),global_step = current_step)

for key in lossdir:
	loss_mean = sum(lossdir[key])/len(lossdir[key])
	result.write(key+','+str(loss_mean))
	result.write('\n')

print max(threshold_b), max(threshold_a)
