import sys
import os
import numpy as np 
import tensorflow as tf 
import keras
from keras import backend as K
from keras.regularizers import l1, l2
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten, Dropout
from keras.layers import Lambda
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
import pickle
from time import time
# from metrics import precision_k_curve,recall_k_curve,ndcg_k_curve
from utils import precision_k_curve, recall_k_curve,hr_k_curve, ndcg_k_curve, cos_sim, average_precision
from tensorflow.contrib import rnn
from tensorflow.python.ops import control_flow_ops  
from tensorflow.python.training import moving_averages  
from tensorflow.python.training.moving_averages import assign_moving_average
from sklearn.cluster import KMeans
from sklearn.preprocessing import label_binarize


class Model():

	def __init__(self,batch_size, num_user,	num_item,num_user_group,
				hidden_size,
				layers = [20,10,5]):
		print("Building model...")
		self.num_user = num_user
		self.num_item = num_item
		self.num_user_group = num_user_group
		# self.embedding_size = embedding_size
		self.hidden_size = hidden_size

		self.user_input = tf.placeholder(tf.int32,shape=[None])
		self.item_input = tf.placeholder(tf.int32, shape=[None])
		self.clustering = tf.placeholder(tf.bool)
		self.user_labels = tf.placeholder(tf.int32,shape=[self.num_user])
		# self.user_labels = tf.placeholder(tf.float32, shape=[None, self.num_user_group])


		self.prediction = tf.placeholder(tf.float32, shape=[None,1])
		self.user_centroids = tf.placeholder(tf.float32, shape=[self.num_user_group, self.hidden_size])



		self.user_embedding = tf.Variable(
			tf.random_uniform([num_user, hidden_size],-1.0,1.0))

		self.user_biase_embedding = tf.Variable(
			tf.random_uniform([num_user, hidden_size], -1.0, 1.0))


		self.item_embedding = tf.Variable(
			tf.random_uniform([num_item, hidden_size],-1.0,1.0))

		# self.user_group_embedding = tf.Variable(
		# 	tf.random_uniform([num_user_group, hidden_size],-1.0,1.0))
		
		one_hot_matrix = np.eye(self.num_user_group).astype(np.float32)


		user_latent = tf.nn.embedding_lookup(self.user_embedding, self.user_input)
		user_biase_latent = tf.nn.embedding_lookup(self.user_biase_embedding, self.user_input)

		item_latent = tf.nn.embedding_lookup(self.item_embedding, self.item_input)
		# user_group_latent = tf.nn.embedding_lookup(self.user_group_embedding, self.user_input)
		user_group_labels = tf.nn.embedding_lookup(self.user_labels, self.user_input)
		user_group_one_hot = tf.nn.embedding_lookup(one_hot_matrix, user_group_labels)
		# user_group_latent = tf.nn.embedding_lookup(self.user_group_embedding, user_group_labels)
		# user_group_latent = tf.matmul(self.user_labels, self.user_centroids)
		user_group_latent = tf.nn.embedding_lookup(self.user_centroids, user_group_labels)



		vector2 = tf.concat([user_latent, user_biase_latent], axis=1)
		cluster_layers = [64,128]
		for i in range(len(cluster_layers)):
			hidden = Dense(cluster_layers[i], activation='relu',kernel_initializer = 'lecun_uniform',name='v2_ui_hidden_' + str(i))
			vector2 = hidden(vector2)
		self.logits_cluster = tf.matmul(vector2, tf.transpose(self.user_centroids))
		# self.loss_cluster = tf.nn.softmax_entropy_with_losits()
		self.loss_cluster = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.logits_cluster, labels = user_group_one_hot))



		sim4 = tf.reduce_sum(tf.multiply(item_latent, user_group_latent), axis=1, keep_dims = True)
		vector4 = tf.concat([item_latent, user_group_latent, sim4, tf.multiply(item_latent, user_group_latent)], axis=1)
		# vector4 = tf.layers.batch_normalization(vector4)


		sim1 = tf.reduce_sum(tf.multiply(user_latent, item_latent), axis=1, keep_dims = True)
		vector1 = tf.concat([user_latent, item_latent,sim1, tf.multiply(user_latent, item_latent) ], axis=1)
		# vector1 = tf.layers.batch_normalization(vector1)




		for i in range(len(layers)):
			hidden = Dense(layers[i], activation='relu',kernel_initializer = 'lecun_uniform',name='v1_ui_hidden_' + str(i))
			vector1 = hidden(vector1)
		self.logits_1 = Dense(1, kernel_initializer='lecun_uniform', name = 'prediction')(vector1)

		for i in range(len(layers)):
			hidden = Dense(layers[i], activation='relu',kernel_initializer = 'lecun_uniform',name='v4_ui_hidden_' + str(i))
			vector4 = hidden(vector4)
		self.logits_4 = Dense(1, kernel_initializer='lecun_uniform', name = 'prediction')(vector4)



		self.logits = tf.cond(self.clustering, lambda: self.logits_4+self.logits_1, lambda: self.logits_1)
		# self.logits = self.logits_1+self.logits_4


		self.pred = tf.nn.sigmoid(self.logits)



		print('haha\n')


		# self.loss = tf.reduce_mean(tf.sigmoid_cross_entropy_with_logits(logits,prediction))	
		self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.logits, labels = self.prediction))



		# reg_error = tf.contrib.layers.l1_regularizer(0.001)()
		# reg_error = self.regularize(f_matrix, self.user_input,  user_embedding)
		# reg_error = self.regularize(f_matrix, f_matrix_1, self.user_input, user_embedding, self.item_adj, self.item_input, item_embedding)
		# self.cost = self.loss+0.01*reg_error
		reg_error = tf.nn.l2_loss(self.user_embedding)\
					+tf.nn.l2_loss(self.item_embedding)\
					+tf.nn.l2_loss(self.user_biase_embedding)
					# +tf.nn.l2_loss(self.user_group_embedding)#+tf.nn.l2_loss(author_embedding)

		# user_group_reg = tf.nn.l2_loss(self.user_group_embedding - self.user_centroids)+\
		# 					tf.nn.l2_loss(user_latent - user_group_latent)

		# reg_error = tf.cond(self.clustering, lambda:reg_error+float(sys.argv[3])*user_group_reg, lambda: reg_error)
		reg_error = tf.cond(self.clustering, lambda:reg_error+self.loss_cluster, lambda:reg_error)
		# reg_error = reg_error+user_group_reg

		self.cost = self.loss+\
					0.0001*reg_error+\
					0.005*self.regularize(self.user_input,  self.user_embedding)
		print('hahah1\n', self.cost.shape)

		self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.cost)
		# self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)



	def regularize(self, user_input, user_embedding):


		# f_matrix = self.load_friend().astype(np.float32)
		f_matrix = np.load('s_matrix.npy').astype(np.float32)

		batch_f_matrix = tf.nn.embedding_lookup(f_matrix, user_input)
		sum_bfm = tf.reduce_sum(batch_f_matrix, axis = 1, keep_dims = True)
		n_sum_bfm = batch_f_matrix/(sum_bfm+10e-10)
		ref_f = tf.matmul(n_sum_bfm, user_embedding)




		batch_embedding = tf.nn.embedding_lookup(user_embedding, user_input)
		reg = tf.reduce_sum(tf.square(batch_embedding - ref_f))
		return reg



	def save_friend(self,f_matrix):
		fr = open('friends.pkl', 'wb')
		data = {}
		data['num_user'] = len(f_matrix)
		f_dic = dict([(i, np.where(f_matrix[i]>0)[0]) for i in range(len(f_matrix)) if len(np.where(f_matrix[i]>0)[0])>0])
		data['f_dic'] = f_dic
		pickle.dump(data, fr)
		fr.close()

	def load_friend(self):
		print('social reg loading...')
		fr = open('friends.pkl','rb')
		data = pickle.load(fr)
		fr.close()

		num_user = data['num_user']
		f_dic = data['f_dic']
		f_matrix = np.zeros((num_user, num_user))
		for i in f_dic.keys():
			f_matrix[i][f_dic[i]] = 1
		return f_matrix


class Data_Loader():
	def __init__(self, batch_size):
		print("data loading...")
		# pickle_file = open("data.pkl",'rb')
		pickle_file = open('musical.pkl','rb')



		self.data = pickle.load(pickle_file)
		self.R_m = self.data['ratings']
		# self.F_matrix = self.data['friends']
		self.num_user = self.data['num_user']
		self.num_item = self.data['num_item']
		self.batch_size = batch_size

		# self.f_matrix = np.zeros((self.num_user, self.num_item))


		# social = self.data['friends']
		# self.f_matrix = np.zeros((self.num_user, self.num_user))
		# for key in social:
		# 	for u in social[key]:
		# 		self.f_matrix[u][social[key]] = 1
		





	def reset_data(self):

		print("resetting data...")
		u_input = self.data['train_user'][:]
		i_input = self.data['train_item'][:]
		item_num = self.data['num_item']
		ui_label = self.data['train_label'][:]
		negative_samples_num = 5
		for u in set(u_input):
			all_item = range(item_num)
			# positive = np.array(self.data['train_item'])[np.where(np.array(self.data['train_user'])==u)[0]]
			missing_values = list(set(all_item)-set(self.R_m[u]))
			# missing_values = list(set(all_item) - set(positive))

			u_input.extend([u]*negative_samples_num)
			negative_samples = np.random.choice(missing_values,negative_samples_num, replace=False)
			i_input.extend(list(negative_samples))
			ui_label.extend([0]*negative_samples_num)

		p_index = np.random.permutation(range(len(u_input)))
		self.u_input = np.array(u_input)[p_index]
		self.i_input = np.array(i_input)[p_index]
		self.ui_label = np.array(ui_label)[p_index]
		self.train_size = len(u_input)



	def reset_pointer(self):
		self.pointer = 0

	def next_batch(self):
		start = self.pointer*self.batch_size
		end = (self.pointer+1)*self.batch_size

		self.pointer+=1
		# return self.u_input[start:end], self.i_input[start:end], self.ui_label[start:end]
		item_index = self.i_input[start:end]
		user_index = self.u_input[start:end]



		return self.u_input[start:end],\
		self.i_input[start:end],\
		self.ui_label[start:end]





def get_data(user,item,data_loader):
	data = data_loader.data 
	# c_item = range(data['num_item'])
	num_item = data['num_item']
	negative_items = 100
	u = [user]*negative_items
	i = [item]+np.random.randint(0,num_item,(negative_items-1)).tolist()
	ui_label = [1]+[0]*(negative_items-1)
	pmtt = np.random.permutation(negative_items)
	return np.array(u),\
			np.array(i)[pmtt],\
			np.array(ui_label)[pmtt]




def test(data_loader, model):
	with tf.Session() as sess:
		# checkpoint_dir = './model_a/'
		# checkpoint_dir = sys.argv[2]
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		saver = tf.train.Saver(tf.global_variables())
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print (' [*] Loaded parameters success!!!')
		else:
			print (' [!] Loaded parameters failed...')

		res_matrix = [[],[],[]]
		count = 0
		max_k=10
		metrics_num = 3
		f = [precision_k_curve,hr_k_curve,ndcg_k_curve]
		clustering = True
		user_labels = np.load(checkpoint_dir+'user_labels.npy')
		user_centroids = np.load(checkpoint_dir+'user_centroids.npy')
		# user_centroids = np.random.random((model.num_user_group, model.hidden_size))
		# user_labels = np.random.randint(0,model.num_user_group, [model.num_user])


		for user,item in zip(data_loader.data['test_user'], data_loader.data['test_item']):
			u,i,ui_label = get_data(user,item,data_loader)

			y_pred = sess.run([model.pred], feed_dict = {model.user_input:u,
														model.item_input:i,
														model.clustering:clustering,
														model.user_labels:user_labels,
														model.user_centroids: user_centroids,
														model.prediction:ui_label.reshape((-1,1))})
			# print(type(y_pred[0]), len(y_pred[0]), 'uqjwen')
			# y_pred = 1.0/(1+np.exp(-y_pred[0]))
			for i in range(metrics_num):
				res = f[i](ui_label.flatten(),y_pred[0].flatten(),max_k)
				# res_matrix[u][i] = res[:]
				res_matrix[i].append(res[:])
			# print (np.array(res_matrix).shape)
			count+=1
			if count%3000==0:
				print (np.mean(np.array(res_matrix),axis=1))
			sys.stdout.write("\ruser: "+str(count))
			sys.stdout.flush()
		print (np.mean(np.array(res_matrix),axis=1))

		# print(np.array(res_matrix).shape)
		
		res = np.mean(np.array(res_matrix), axis=1).T
		# res = np.concatenate([res,var],axis=1)
		np.savetxt(checkpoint_dir+'res.dat',res, fmt='%.5f', delimiter='\t')
			
			

# def user_clustering()
### return cluster centroids and cluster id for each data point
def user_clustering(iterations,T1,user_embedding, user_centroids, user_labels, num_group):
	T2 = 20
	if iterations+1 == T1:
		###first time to cluster
		kmeans = KMeans(n_clusters = num_group, max_iter = T2).fit(user_embedding)
		return kmeans.cluster_centers_, kmeans.labels_
	else:
		kmeans = KMeans(n_clusters = num_group, init = user_centroids, max_iter = T2).fit(user_embedding)
		return kmeans.cluster_centers_, kmeans.labels_



def sample(u,i):
	user = []
	item = []
	for usr,itm in zip(u,i):
		rand = np.random.random()
		if rand<0.5:
			user.append(usr)
			item.append(itm)
	return user,item
			

def val(data_loader, sess, model, user_centroids, user_labels, tv_user, tv_item):
	res_matrix = [[],[],[]]
	max_k=10
	metrics_num = 3
	f = [precision_k_curve,hr_k_curve,ndcg_k_curve]


	# tv_user,tv_item = sample(data_loader.data['val_user'], data_loader.data['val_item'])
	# user_centroids = np.random.random((model.num_user_group, model.hidden_size))
	# user_labels = np.random.randint(0,model.num_user_group, [model.num_user])
	clustering = True

	for user,item in zip(tv_user, tv_item):
		u,i,ui_label = get_data(user,item,data_loader)

		y_pred = sess.run([model.pred], feed_dict = {model.user_input:u,
													model.item_input:i,
													model.clustering:clustering,
													model.user_labels:user_labels,
													model.user_centroids: user_centroids,
													model.prediction:ui_label.reshape((-1,1))})

		for i in range(metrics_num):
			res = f[i](ui_label.flatten(),y_pred[0].flatten(),max_k)
			# res_matrix[u][i] = res[:]
			res_matrix[i].append(res[:])
		# print (np.array(res_matrix).shape)
		
	res = np.mean(np.array(res_matrix), axis=1).T
	return res[-1,1]


def train(batch_size, epochs,data_loader, layers, model):
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		# sess.run(tf.assign(model.word_embedding, data_loader.init_embedding))
		saver = tf.train.Saver(tf.global_variables())
		# checkpoint_dir = 'model'
		# checkpoint_dir = sys.argv[2]
		# checkpoint_dir = './'+sys.argv[0].split('.')[0]+'/'

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print (' [*] Loaded parameters success!!!')
		else:
			print (' [!] Loaded parameters failed...')

		epochs_1 = 5
		epochs_2 = 50
		# data_loader.reset_data()
		clustering = False
		user_centroids = np.random.random((model.num_user_group, model.hidden_size))
		user_labels = np.random.randint(0,model.num_user_group, [model.num_user])
		tv_user,tv_item = sample(data_loader.data['val_user'], data_loader.data['val_item'])

		item_embedding = np.load('./neumf_so/item_embedding.npy')
		user_embedding = np.load('./neumf_so/user_embedding.npy')
		sess.run(tf.assign(model.item_embedding, item_embedding))
		sess.run(tf.assign(model.user_embedding, user_embedding))

		best_hr_10 = 0
		for i in range(epochs_1):
			data_loader.reset_data()
			for e in range(epochs_2):
				# data_loader.reset_data()
				total_batch = int(data_loader.train_size/batch_size)

				data_loader.reset_pointer()
				for b in range(total_batch):
					iterations = i*epochs_2*total_batch+e*total_batch+b 

					u_input,i_input, ui_label = data_loader.next_batch()

					# if(iterations+1)%100==0:
					# 	user_embedding = sess.run([model.user_embedding])


					train_loss,_ = sess.run([model.cost, model.train_op], feed_dict={model.user_input:u_input,
																						model.item_input:i_input,
																						model.clustering:clustering,
																						model.user_labels:user_labels,
																						model.user_centroids: user_centroids,
																						model.prediction:ui_label.reshape((-1,1))})


					T1 = 100
					if (iterations+1)%T1 == 0:
						clustering = True
						user_embedding = sess.run(model.user_embedding)
						# print(user_embedding.shape)
						user_centroids, user_labels =  user_clustering(iterations,T1,
																			user_embedding, 
																			user_centroids, 
																			user_labels, 
																			model.num_user_group)
						# _ = sess.run([model.group_cluster], feed_dict={model.clustering:True})
						##performance clustering here
					sys.stdout.write('\r {}/{} epoch, {}/{} batch. train_loss:{}'.\
									format(i,e, b, total_batch, train_loss))
					sys.stdout.flush()


					if (iterations)%1000==0:
						hr_10 = val(data_loader, sess, model, user_centroids, user_labels, tv_user,tv_item)
						if hr_10>best_hr_10:
							print('\n',hr_10)
							best_hr_10 = hr_10
							saver.save(sess, checkpoint_dir+'model.ckpt', global_step = iterations)
							sys.stdout.flush()
							np.save(checkpoint_dir+'user_centroids', user_centroids)
							np.save(checkpoint_dir+'user_labels', user_labels)
		# saver.save(sess, checkpoint_dir+'model.ckpt', global_step = iterations)



checkpoint_dir = './'+sys.argv[0].split('.')[0]+'_'+sys.argv[2]+'/'

if __name__ == '__main__':
	batch_size = 256
	if len(sys.argv)>1 and sys.argv[1] == 'test':
		batch_size = 100

	epochs = 100
	data_loader = Data_Loader(batch_size = batch_size)
	layers = eval("[64,16]")
	num_user_group = int(sys.argv[2])


	model = Model(
				batch_size = batch_size,
				num_user = data_loader.num_user,
				num_item = data_loader.num_item,
				num_user_group = num_user_group,
				hidden_size = 128,
				layers = layers)
	if sys.argv[1]=="test":
		test(data_loader, model)
	else:
		train(batch_size, epochs, data_loader, layers, model)
