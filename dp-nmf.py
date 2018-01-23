#Author: Satwik Bhattamishra

import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
from pnmf import PNMF
import numpy.linalg as LA

def dpnmf(hidden_units= 500, prob=0.4, lambd= 0.4, alpha= 0.2, beta= 0.2):

###### Denoising Autoencoder ######

# Parameter Declaration
	learning_rate = 0.001
	batch_size = 50
	n_epochs = 50

# Variables/Tensors Declaration

	x = tf.placeholder(tf.float32, [None, 784], name='x')

	dfro = tf.placeholder(tf.float32, [None, 500], name='dfro')
	x_ = tf.add(x ,np.random.normal(loc= 0.0, scale=prob , size= (batch_size, 784) ))
	ind = tf.Variable(0, tf.int32)

	n_inp = 784
	n_out = hidden_units

	A = tf.Variable(tf.random_uniform([n_inp, n_out], -1.0 / np.sqrt(n_inp), 1.0 / np.sqrt(n_inp)) ,dtype=tf.float32 )
	b = tf.Variable(tf.truncated_normal([n_out], dtype=tf.float32))
	A_ = tf.Variable(tf.random_uniform([n_out, n_inp], -1.0 / np.sqrt(n_inp), 1.0 / np.sqrt(n_inp)) ,dtype=tf.float32 )
	b_ = tf.Variable(tf.truncated_normal([n_inp], dtype=tf.float32))

	z = tf.nn.sigmoid(tf.matmul(x_ , A) + b)

	y = tf.nn.sigmoid(tf.matmul(z , A_) + b_)


	cost = tf.reduce_mean(-tf.reduce_sum(x * tf.log(tf.clip_by_value(y ,1e-10,1.0))) )	#Cross Entropy Loss

# Manual Gradient Computation

	lout = tf.subtract(y,x)
	lh = tf.multiply(tf.multiply(tf.matmul(lout, A), z) , (tf.subtract(1.0,z)) )

	lb = lh
	lb_ = lout

	grad_A = tf.add(tf.matmul(tf.transpose(x_) , lh), tf.matmul(tf.transpose(lout), z ))

	grad_b = tf.reduce_mean(lb, axis=0)
	grad_b_ = tf.reduce_mean(lb_, axis=0)

	new_A = A.assign(A - learning_rate * grad_A)
	new_A_ = A_.assign(tf.transpose(A))

	new_b = b.assign(b - learning_rate * grad_b )
	new_b_ = b_.assign(b_ - learning_rate * grad_b_ )
	

	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	mean_img = np.mean(mnist.train.images, axis=0)


	saver = tf.train.Saver()



###### Denoising Autoencoder Training ######

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	# saver.restore(sess, "./weights/da.cpkt")
	print "Pretraining DA..."

	for epoch_i in range(n_epochs):
		avg_cost = 0
		batches= mnist.train.num_examples // batch_size
		for batch_i in range(batches):
			batch_xs, _ = mnist.train.next_batch(batch_size)

			_, _, _, _, ce = sess.run([new_A, new_A_, new_b, new_b_, cost], feed_dict={x: batch_xs})
			# _,ce = sess.run([optimizer, cost], feed_dict={x: train})

			avg_cost += ce / batches

		print(epoch_i, avg_cost)



	save = saver.save(sess, "./weights/da.ckpt")


###### Finetuning and factor computation ######

# Parameter Declaration

	n_iter = 50			# Number of total iterations
	nd_iter = 30		# Number subiterations for Mul-update rules
	rank = 10 			# Rank for NMF

	z = tf.nn.sigmoid(tf.matmul(x , A) + b)			# clean input from further computations

	y = tf.nn.sigmoid(tf.matmul(z , A_) + b_)

# Computing f(V), reduced input

	train_xs = mnist.train.images
	fV = sess.run(z, feed_dict={x: train_xs})
	W = np.random.random((fV.shape[0], rank))
	H = np.random.random((rank, fV.shape[1]))

	np.save('encodings/mnist_'+str(hidden_units), fV)


	cost = tf.reduce_mean(-tf.reduce_sum(x * tf.log(tf.clip_by_value(y ,1e-10,1.0))) )

# Computing Custom Gradients 

	lout = tf.subtract(y,x)
	lh = tf.multiply(tf.multiply(tf.matmul(lout, A), z) , (tf.subtract(1.0,z)) )

	lb = lh
	lb_ = lout

	frob_norm = 2*(fV - np.dot(W,H))

	grad_A1 = tf.add(tf.matmul(tf.transpose(x), lh), tf.matmul(tf.transpose(lout), z ))
	grad_A2 = tf.matmul(tf.transpose(x), tf.multiply(z, tf.subtract(1.0, z)))*2*tf.reduce_mean(dfro, axis=0)
	grad_A  = lambd*grad_A1 + grad_A2

	grad_b1 = tf.reduce_mean(lb, axis=0)
	grad_b2 = tf.reduce_mean(tf.multiply(z , tf.subtract(1.0, z))*dfro, axis=0)
	grad_b = lambd*grad_b1 + grad_b2

	grad_b_ = tf.reduce_mean(lb_, axis=0)

	new_A = A.assign(A - learning_rate * grad_A)
	new_A_ = A_.assign(A_ - learning_rate * tf.transpose(grad_A1) )

	new_b = b.assign(b - learning_rate * grad_b )
	new_b_ = b_.assign(b_ - learning_rate * grad_b_ )



	print "Finetuning..."

	for i in range(n_iter):

		pnmf = PNMF(fV, W=W, H=H, rank=rank)
		pnmf.compute_factors(max_iter= nd_iter, alpha= alpha, beta= beta)
		W = pnmf.W
		H= pnmf.H

		avg_cost = 0
		batches= mnist.train.num_examples // batch_size
		for batch_i in range(batches):
			batch_xs, _ = mnist.train.next_batch(batch_size)
			frob_errors = frob_norm[batch_size*batch_i :batch_size*batch_i  + batch_size ]
			_, _, _, _, nind, ce = sess.run([new_A, new_A_, new_b, new_b_, new_ind, cost], feed_dict={x: batch_xs, dfro: frob_errors})
			
			frob_norm = 2*(fV - np.dot(W,H))
			avg_cost += ce / batches

		frob_error =  LA.norm(fV-np.dot(W,H)) 
		total_loss = lambd * avg_cost + frob_error


		print str(i)+ " : " + str(lambd)+"*"+str(avg_cost) + "+"+str(frob_error)+"="+str(total_loss)




if __name__ == '__main__':
	dpnmf()
