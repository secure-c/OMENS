# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
def dyconv(
		inputs,
		inputs_size, #[batch,heigth,weigth,1]
		W,
		filter_size, #[filter_height,filter_weigth,1,num_filters]
		Wg,			 #[num_filters,weigth,1]
		bg):		 #[num_filters]
	heigth = inputs_size[1]
	weigth = inputs_size[2]
	conv_num = heigth
	conv = []
	gate_i = []
	for i in range(filter_size[3]):
		Wg_i = Wg[i,:,:]
		bg_i = bg[i]
		W_i = W[:,:,0,i] #[filter_heigth,filter_weigth]
		conv_temp = []
		gate_j = []
		for j in range(conv_num):
			#x_conv's shape is [batch,filter_heigth,weigth]
			x_conv = inputs[:,j:j+filter_size[0],:,0]
			W_j_sum = 0
			gate_k = []
			for k in range(filter_size[0]):
				x_line = x_conv[:,k,:] #[batch,weigth]
				gate = tf.nn.sigmoid(tf.matmul(x_line,Wg_i) + bg_i, name = "gate") #[batch,1]
				W_jk = tf.reshape(W_i[k], [weigth,1], name = "W_jk") #[weigth,1]
				W_jk_sum = tf.matmul(x_line, W_jk, name = "W_jk_sum") #[batch,1]
				W_j_sum += gate * W_jk_sum #[batch,1]
				gate_k.append(gate) #[filter_heigth,batch,1]
			conv_temp.append(W_j_sum)
			gate_j.append(gate_k) #[conv_num,filter_heigth,batch,1]
		W_j = tf.concat(conv_temp, 1, name = "W_j")
		W_j = tf.expand_dims(W_j,-1)
		conv.append(W_j)
		gate_i.append(gate_j) #[num_filters,conv_num,filter_heigth,batch,1]
	conv = tf.concat(conv, 2) #[batch,conv_num,num_filters]
	conv = tf.expand_dims(conv,2) #[batch,conv_num,1,num_filters]

	return conv, gate_i

	
