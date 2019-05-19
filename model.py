import numpy as np
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers import Dense, Dropout, Embedding, Input
from keras.layers.core import Activation, RepeatVector, Permute, Reshape
from keras.layers.recurrent import LSTM

class Attention(Layer):
	def __init__(self, **kwargs):
		self.supports_masking = True
		self.trainable = True
		self.init_stdev = 0.01
		super(Attention, self).__init__(**kwargs)

	def build(self, input_shape):
		init_val_v = (randm.randn(input_shape[2]) * self.init_stdev).astype(K.floatx())
		self.att_v = K.variable(init_val_v, name='att_v')
		init_val_W = (randm.randn(input_shape[2], input_shape[2]) * self.init_stdev).astype(K.floatx())
		self.att_W = K.variable(init_val_W, name='att_W')
		self.trainable_weights = [self.att_v, self.att_W]
	
	def call(self, x, mask=None):
		y = K.dot(x, self.att_W)
		weights = K.tf.tensordot(self.att_v, K.tanh(y), axes=[[0], [2]])
		weights = K.softmax(weights)
		out = x * K.permute_dimensions(K.repeat(weights, x.shape[2]), [0, 2, 1])
		out = K.sum(out, axis=1)
		return K.cast(out, K.floatx())

	def get_output_shape_for(self, input_shape):
		return (input_shape[0], input_shape[2])
	
	def compute_mask(self, x, mask):
		return None
	
	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[2])

def read_emb(emb_path):
	embeddings = {}
	with open(emb_path, 'r') as f:
		for line in f:
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype=K.floatx())
			embeddings[word] = coefs
	f.close()
	return embeddings

def get_emb_matrix_given_vocab(vocab, source_emb_matrix, dest_emb_matrix, emb_dim):
	counter = 0.
	for word, index in enumerate(vocab): #vocab.iteritems() in python2
		try:
			if index == 0:
				dest_emb_matrix[index] = np.zeros(emb_dim, dtype=K.floatx())
			else:
				dest_emb_matrix[index] = source_emb_matrix[word]
			counter += 1
		except KeyError:
			pass
	return dest_emb_matrix


def create_model(args, vocab, vocab_size, grs_size):
	embeddings = read_emb(args.emb_path)
	embedding_matrix = np.random.normal(loc=0.0, scale=0.01, size=(vocab_size, args.emb_dim))
	emb_main = Embedding(vocab_size, args.emb_dim, mask_zero=True, weights=[get_emb_matrix_given_vocab(vocab, embeddings, embedding_matrix, args.emb_dim)]) 

	sequence = Input(shape=(None, None, ),
              dtype='int32', name='main_input')
	
	embedded_sent = TimeDistributed(emb_main)(sequence)
	embedded_sent = Dropout(args.dropout)(embedded_sent)
	rnn_sent = TimeDistributed(Bidirectional(LSTM(args.rnn_dim, return_sequences=True),merge_mode='concat'))(embedded_sent)
	#predict gr roles
	n_tags = grs_size
	grs = TimeDistributed(TimeDistributed(Dense(n_tags, activation="softmax")), name='gr_scores')(rnn_sent)
	rnn_sent = TimeDistributed(Attention())(rnn_sent)

	merged = Bidirectional(LSTM(args.rnn_dim, return_sequences=True),merge_mode='concat')(rnn_sent)
	merged = Dropout(args.dropout)(merged)
	mot = Attention()(merged)
	dense = Dense(1)(mot)
	score_gc = Activation('sigmoid', name='score')(dense)
	model = Model(input=sequence, output=[score_gc, grs])
	if args.model_path != "":
		model.load_weights(args.model_path, by_name=True)
	model.summary()
	
	return model
