import tensorflow as tf

#	Lovingly paraphrased from:
#	https://github.com/tensorflow/models/blob/master/research/nst_blogpost/4_Neural_Style_Transfer_with_Eager_Execution.ipynb

#	Content loss is mean squared error
def get_content_loss(base_content, target):
	return tf.reduce_mean(tf.square(base_content - target))

def gram_matrix(input_tensor):
	# We make the image channels first 
	channels = int(input_tensor.shape[-1])
	a = tf.reshape(input_tensor, [-1, channels])
	n = tf.shape(a)[0]
	gram = tf.matmul(a, a, transpose_a=True)
	return gram / tf.cast(n, tf.float32)

def get_style_loss(gram_style, gram_target):
	# We scale the loss at a given layer by the size of the feature map and the number of filters
	return tf.reduce_mean(tf.square(gram_style - gram_target))# / (4. * (channels ** 2) * (width * height) ** 2)