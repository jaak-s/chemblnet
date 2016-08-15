import vbutils as vb
import tensorflow as tf
import numpy as np

x = tf.Variable(np.array([[1., 2.], [3., 4.], [5., 6.]]), dtype=tf.float32)

x_indices  = tf.placeholder(tf.int64)
x_shape    = tf.placeholder(tf.int64)
x_ids_val  = tf.placeholder(tf.int64)
sp_ids  = tf.SparseTensor(x_indices, x_ids_val, x_shape)

eexp = vb.embedding_lookup_sparse_sumexp(x, sp_ids)

bx_indices = np.array([[0], [1], [1]])
bx_ids_val = np.array([0, 1, 2])
bx_shape   = [0, 0]

sess = tf.Session()
sess.run(tf.initialize_all_variables())
sess.run(tf.nn.embedding_lookup_sparse(x, sp_ids, None, combiner = "sum"),
         feed_dict={x_indices: bx_indices,
                    x_shape:   bx_shape,
                    x_ids_val: bx_ids_val})

sess.run(tf.nn.embedding_lookup_sparse(x, sp_ids, None, combiner = "sum"),
         feed_dict={x_indices: bx_indices,
                    x_shape:   bx_shape,
                    x_ids_val: bx_ids_val,
                    x: np.exp(np.array([[1., 2.], [3., 4.], [5., 6.]]))
                    })

sess.run(eexp,
         feed_dict={x_indices: bx_indices,
                    x_shape:   bx_shape,
                    x_ids_val: bx_ids_val})

