import tensorflow as tf
from tensorflow.python.framework.ops import colocate_with

class SGLD(tf.train.Optimizer):
    def __init__(self,
                 learning_rate,
                 use_locking=False,
                 name="SGLD"):
        super(SGLD, self).__init__(use_locking, name)
        self._opt = tf.train.GradientDescentOptimizer(learning_rate)
        self._learning_rate = learning_rate

    def compute_gradients(self, *args, **kwargs):
        return self._opt.compute_gradients(*args, **kwargs)

    def get_slot(self, *args, **kwargs):
        return self._opt.get_slot(*args, **kwargs)

    def get_slot_names(self, *args, **kwargs):
        return self._opt.get_slot_names(*args, **kwargs)

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        with tf.name_scope(name, self._name) as name:
            update_op = self._opt.apply_gradients(
                              grads_and_vars, global_step=global_step)
            add_noise_ops = []
            with tf.control_dependencies([update_op]):
                for grad, var in grads_and_vars:
                    if grad is None:
                        continue
                    with tf.name_scope("sgld_noise_" + var.op_name):
                        if isinstance(grad, ops.Tensor):
                            add_noise_ops.append(self._noise_dense(var))
                        else:
                            add_noise_ops.append(self._noise_sparse(grad, var))
            ## running combined op
            return tf.group(*([update_op] + add_noise_ops), name=name)

    def _noise_dense(self, var):
        updated_var_value = var._ref()  # pylint: disable=protected-access
        noise = tf.random_normal(shape = tf.shape(var), stddev = tf.sqrt(self._learning_rate))
        with colocate_with(var):
            return var.assign_add(noise)

    def _noise_sparse(self, grad, var):
        ## TODO: add noise
        pass
