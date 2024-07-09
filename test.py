# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from sklearn import metrics
from time import time
from Queue import Queue
import os
import itertools
import sys
import data_helpers
from CNN import CNN
import networkx as nx

#fs = sys.argv[1]
fs = '1'
BATCH_SIZE = 1
NUM_CLASSES = 2
FILTER_SIZES = 8
#USERNAME = ['fileName32_test.txt_add']
#USERNAME = ['fileName32_1.txt_test_evasion']
USERNAME = ['fileName26_1.txt_test',
		'fileName30_1.txt_test',
		'fileName32_1.txt_test']
#USERNAME = ['fileName26_1.txt_test']
IS_RATE = True
IS_NEGATION = False
#LENGTH = 201
POSITION_RATE = 0.8

MODEL_SAVE_PATH = "./to/model/"
MODEL_NAME = "model.ckpt"
#MODELB_SAVE_PATH = "./to/modelB/"
#MODELB_NAME = "modelb.ckpt"

data_list = []
label_list = []
date_list = []
user_list = []
labelcontext_list = []
datecontext_list = []


for i in USERNAME:
	DATA_PATH = 'insider_data/dataset/'+i+'_add_onehot'
	LABEL_PATH = 'insider_data/dataset/'+i+'_add_label'
	data_i, label_i, date_i, l_i = data_helpers.data_load_1(DATA_PATH, LABEL_PATH, NUM_CLASSES,i)
	NUM_DIM = len(data_i[0])
	x_i, lc_i, dc_i = data_helpers.generating_x_l(np.array(data_i), np.array(l_i), np.array(date_i), FILTER_SIZES,NUM_DIM)
	data_list = data_list+x_i
	labelcontext_list = labelcontext_list+lc_i
	datecontext_list = datecontext_list+dc_i
	label_list = label_list+label_i
	date_list = date_list+date_i

data = np.array(data_list)
labelcontext = np.array(labelcontext_list)
datecontext = np.array(datecontext_list)
label = np.array(label_list)
date = np.array(date_list)
#print label.shape
#print data.shape
representationfile_b = open('./representation_b.csv','w')
representationfile_a = open('./representation_a.csv','w')

threshold_a = 0.246
threshold_b = 0.5

alertevents = []
f_add = open('./add_events.csv')
#add_events = f_add.readline().rstrip().split(',')
add_events = []

checkpoint_file = tf.train.latest_checkpoint(MODEL_SAVE_PATH)
#checkpoint_fileb = tf.train.latest_checkpoint(MODELB_SAVE_PATH)

graph = tf.Graph()

with graph.as_default():
	sess = tf.Session()
	with sess.as_default():
		saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
		saver.restore(sess, checkpoint_file)
		input_x = graph.get_operation_by_name("input_x").outputs[0]
		batch_size = graph.get_operation_by_name("batch_size").outputs[0]
		dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
		representation_b = graph.get_operation_by_name("h_p").outputs[0]
		representation_a = graph.get_operation_by_name("h_n").outputs[0]
		gatep = graph.get_operation_by_name("gatep").outputs[0]
		gaten = graph.get_operation_by_name("gaten").outputs[0]
		predictions = graph.get_operation_by_name("CNN_output/softmax").outputs[0]
		representation_b_input = graph.get_operation_by_name("representation_b").outputs[0]
		representation_a_input = graph.get_operation_by_name("representation_a").outputs[0]
		recontruction_error_b = graph.get_operation_by_name("autoencoderloss_b/autoencoderloss_b").outputs[0]
		recontruction_error_a = graph.get_operation_by_name("autoencoderloss_a/autoencoderloss_a").outputs[0]
		
		attack_num = 0
		benign_num = 0
		benign_detection_num = 0
		attack_detection_num = 0
		#benign_detection_add_num = 0
		add_detection_num = 0

		benign_num_c = 0
		attack_num_c = 0
		benign_detection_num_c = 0
		attack_detection_num_c = 0
		fbenign_detection_num_c = 0
		fattack_detection_num_c = 0
		alertid = 0
		
		start_time = time()

		for i, x_line in enumerate(data):
			x_line = np.array([x_line.tolist()])
			predictions_show, representation_r_b, representation_r_a, gatep_r, gaten_r = sess.run([predictions, representation_b, representation_a, gatep, gaten], {input_x:x_line, batch_size: BATCH_SIZE, dropout_keep_prob: 1.0})
			recontruction_b = sess.run([recontruction_error_b],{representation_b_input:representation_r_b})
			recontruction_a = sess.run([recontruction_error_a],{representation_a_input:representation_r_a})
			if date[i] != '0':
				labelcontext_i = np.r_[labelcontext[i][0:FILTER_SIZES],labelcontext[i][FILTER_SIZES+1:]]
				datecontext_i = np.r_[datecontext[i][0:FILTER_SIZES],datecontext[i][FILTER_SIZES+1:]]
				
				#print date[i], label[i], gatep_r[:,0,0], recontruction_b
				#print date[i], label[i], gaten_r[:,0,0], recontruction_a
				recontruction_b = recontruction_b[0]
				recontruction_a = recontruction_a[0]
				if label[i,0] == 1:
					benign_num += 1
					if recontruction_a <= threshold_a or (recontruction_a > threshold_a and recontruction_b > threshold_b*2):
					#if recontruction_a <= threshold_a:
						alertid += 1		
						alertevents.append((date[i]+'_fal_'+str(alertid),x_line[0,FILTER_SIZES,:]))
						print "false", date[i], label[i], gaten_r[:,0,0], recontruction_a
						print "false", date[i], label[i], gaten_r[:,0,0], recontruction_b

						if date[i] not in add_events:	
							benign_detection_num += 1
						else:
							add_detection_num += 1
						#if date[i] in add_events:
						#	benign_detection_add_num += 1
					for k, l in enumerate(labelcontext_i):
						if l == -1:continue
						elif l == 0:
							benign_num_c += 1
							if gatep_r[:,0,0][k] == 1:
								benign_detection_num_c += 1
						elif gatep_r[:,0,0][k] == 1:
							fbenign_detection_num_c += 1

				else:
					attack_num += 1
					if recontruction_a <= threshold_a or (recontruction_a > threshold_a and recontruction_b > threshold_b*2):
					#if recontruction_a <= threshold_a:
						attack_detection_num += 1
						alertid += 1
						print "true", date[i], label[i], gaten_r[:,0,0], recontruction_a
						print "true", date[i], label[i], gaten_r[:,0,0], recontruction_b
						alertevents.append((date[i]+'_tru_'+str(alertid),x_line[0,FILTER_SIZES,:]))

					for k, l in enumerate(labelcontext_i):
						if l == -1:continue
						elif l == 1:
							attack_num_c += 1
							if gaten_r[:,0,0][k] == 1:
								attack_detection_num_c += 1
						elif gaten_r[:,0,0][k] == 1:
							fattack_detection_num_c += 1
				#graphB = tf.Graph()
				#with graphB.as_default():
					#sessb = tf.Session()
					#with sessb.as_default():
						#story_starttime = time()
						#saverb = tf.train.import_meta_graph("{}.meta".format(checkpoint_fileb))
						#saverb.restore(sessb, checkpoint_fileb)
						#cor_x = graphB.get_operation_by_name("cor_x").outputs[0]
						#score_cp = graphB.get_operation_by_name("score_cp").outputs[0]
					
		end_time = time()
		vent_time = (end_time-start_time) / (attack_num + benign_num)

recall = 1.0*attack_detection_num / attack_num
if attack_detection_num + benign_detection_num != 0:
    precise = 1.0*attack_detection_num / (attack_detection_num + benign_detection_num)
else:
	precise = 0.0
print recall, precise


MODELB_SAVE_PATH = "./to/modelB/"
MODELB_NAME = "modelb.ckpt"
checkpoint_file = tf.train.latest_checkpoint(MODELB_SAVE_PATH)
graphB = tf.Graph()
with graphB.as_default():
	sess = tf.Session()
	with sess.as_default():
		story_starttime = time()
		saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
		saver.restore(sess, checkpoint_file)
		cor_x = graphB.get_operation_by_name("cor_x").outputs[0]
		score_cp = graphB.get_operation_by_name("score_cp").outputs[0]
		
		
		pair_link = []
		group_false_num = 0
		group = []
		graph = nx.Graph()

		for i in range(len(alertevents)):
			if i>1:
				step_list = [(alertevents[i],alertevents[j]) for j in range(i)]
				for pair_i in step_list:
					pair_x = pair_i[0][1]+pair_i[1][1]
					pair_x = np.array([pair_x.tolist()])
					s_cp = sess.run([score_cp], {cor_x:pair_x})
					if s_cp[0][0][1] > 0.5:
						if pair_i[0][0].split('_')[1] == pair_i[1][0].split('_')[1]:
							t1 = int(pair_i[0][0].split('_')[-1])
							t2 = int(pair_i[1][0].split('_')[-1])
							pair_tmp = []
							if t1 < t2:
								pair_tmp = [pair_i[0][0],pair_i[1][0],s_cp[0][0][1]]
							else:
								pair_tmp = [pair_i[1][0],pair_i[0][0],s_cp[0][0][1]]
							pair_link.append(pair_tmp)
							graph.add_edge(pair_tmp[0],pair_tmp[1],score = pair_tmp[2])
							
				for c in nx.connected_components(graph):
					nodeSet = graph.subgraph(c).nodes()
					edgeSet = graph.subgraph(c).edges(data=True)
					score = 0.0
					for edge in edgeSet:
						score += edge[2]['score']
					score = score/len(edgeSet)	
					group.append((nodeSet,score))

		story_endtime = time()
		for c in nx.connected_components(graph):
			nodes = graph.subgraph(c).nodes()
			if len(nodeSet) >= 2:
				for n in nodes:
					if 'fal' in n:
						group_false_num += 1


print 'Number of events: ', str(len(data))
print 'Number of alert events: ', str(len(alertevents))
print 'Time cost of correlation: ', str(story_endtime-story_starttime)
#for one in alertevents:
	#print alertevents
recall = 1.0*attack_detection_num / attack_num
if attack_detection_num + benign_detection_num != 0:
	precise = 1.0*attack_detection_num / (attack_detection_num + benign_detection_num)
else:
	precise = 0.0

if attack_detection_num + group_false_num != 0:
	precise_group = 1.0*attack_detection_num / (attack_detection_num + group_false_num)
else:
	precise_group = 0.0

print recall, precise, precise_group
