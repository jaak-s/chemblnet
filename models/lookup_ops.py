from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops

from tensorflow.python.ops.embedding_ops import embedding_lookup

def embedding_lookup_sparse_sq(params, sp_ids, sp_weights,
                               partition_strategy="mod",
                               name=None,
                               combiner="mean"):
  """Computes squares of embeddings for the given ids and weights.
  This op assumes that there is at least one id for each row in the dense tensor
  represented by sp_ids (i.e. there are no rows with empty features), and that
  all the indices of sp_ids are in canonical row-major order.
  It also assumes that all id values lie in the range [0, p0), where p0
  is the sum of the size of params along dimension 0.
  Args:
    params: A single tensor representing the complete embedding tensor,
      or a list of P tensors all of same shape except for the first dimension,
      representing sharded embedding tensors.
    sp_ids: N x M SparseTensor of int64 ids (typically from FeatureValueToId),
      where N is typically batch size and M is arbitrary.
    sp_weights: either a SparseTensor of float / double weights, or None to
      indicate all weights should be taken to be 1. If specified, sp_weights
      must have exactly the same shape and indices as sp_ids.
    partition_strategy: A string specifying the partitioning strategy, relevant
      if `len(params) > 1`. Currently `"div"` and `"mod"` are supported. Default
      is `"mod"`. See `tf.nn.embedding_lookup` for more details.
    name: Optional name for the op.
    combiner: A string specifying the reduction op. Currently "mean", "sqrtn"
      and "sum" are supported.
      "sum" computes the weighted sum of the embedding results for each row.
      "mean" is the weighted sum divided by the total weight.
      "sqrtn" is the weighted sum divided by the square root of the sum of the
      squares of the weights.

  Returns:
    A dense tensor representing the combined embeddings for the
    sparse ids. For each row in the dense tensor represented by sp_ids, the op
    looks up the embeddings for all ids in that row, multiplies them by the
    corresponding weight, and combines these embeddings as specified.
    In other words, if
      shape(combined params) = [p0, p1, ..., pm]
    and
      shape(sp_ids) = shape(sp_weights) = [d0, d1, ..., dn]
    then
      shape(output) = [d0, d1, ..., dn-1, p1, ..., pm].
    For instance, if params is a 10x20 matrix, and sp_ids / sp_weights are
      [0, 0]: id 1, weight 2.0
      [0, 1]: id 3, weight 0.5
      [1, 0]: id 0, weight 1.0
      [2, 3]: id 1, weight 3.0
    with combiner="mean", then the output will be a 3x20 matrix where
      output[0, :] = (params[1, :]**2 * 2.0 + params[3, :]**2 * 0.5) / (2.0 + 0.5)
      output[1, :] = params[0, :]**2  * 1.0
      output[2, :] = params[1, :]**2  * 3.0
  Raises:
    TypeError: If sp_ids is not a SparseTensor, or if sp_weights is neither
      None nor SparseTensor.
    ValueError: If combiner is not one of {"mean", "sqrtn", "sum"}.
  """
  if combiner not in ("mean", "sqrtn", "sum"):
    raise ValueError("combiner must be one of 'mean', 'sqrtn' or 'sum'")
  if not isinstance(params, list):
    params = [params]
  if not isinstance(sp_ids, ops.SparseTensor):
    raise TypeError("sp_ids must be SparseTensor")
  ignore_weights = sp_weights is None
  if not ignore_weights:
    if not isinstance(sp_weights, ops.SparseTensor):
      raise TypeError("sp_weights must be either None or SparseTensor")
    sp_ids.values.get_shape().assert_is_compatible_with(
        sp_weights.values.get_shape())
    sp_ids.indices.get_shape().assert_is_compatible_with(
        sp_weights.indices.get_shape())
    sp_ids.shape.get_shape().assert_is_compatible_with(
        sp_weights.shape.get_shape())
    # TODO(yleon): Add enhanced node assertions to verify that sp_ids and
    # sp_weights have equal indices and shapes.

  with ops.op_scope(params + [sp_ids], name, "embedding_lookup_sparse_sq") as name:
    segment_ids = sp_ids.indices[:, 0]
    if segment_ids.dtype != dtypes.int32:
      segment_ids = math_ops.cast(segment_ids, dtypes.int32)

    ids = sp_ids.values
    if ignore_weights:
      ids, idx = array_ops.unique(ids)
    else:
      idx = None

    embeddings = embedding_lookup(
        params, ids, partition_strategy=partition_strategy)
    embeddings = math_ops.square(embeddings)
    if not ignore_weights:
      weights = sp_weights.values
      if weights.dtype != embeddings.dtype:
        weights = math_ops.cast(weights, embeddings.dtype)

      # Reshape weights to allow broadcast
      ones = array_ops.fill(
          array_ops.expand_dims(array_ops.rank(embeddings) - 1, 0), 1)
      bcast_weights_shape = array_ops.concat(0, [
          array_ops.shape(weights), ones])

      orig_weights_shape = weights.get_shape()
      weights = array_ops.reshape(weights, bcast_weights_shape)

      # Set the weight shape, since after reshaping to bcast_weights_shape,
      # the shape becomes None.
      if embeddings.get_shape().ndims is not None:
        weights.set_shape(orig_weights_shape.concatenate(
            [1 for _ in range(embeddings.get_shape().ndims - 1)]))

      embeddings *= weights

      if combiner == "sum":
        embeddings = math_ops.segment_sum(embeddings, segment_ids, name=name)
      elif combiner == "mean":
        embeddings = math_ops.segment_sum(embeddings, segment_ids)
        weight_sum = math_ops.segment_sum(weights, segment_ids)
        embeddings = math_ops.div(embeddings, weight_sum, name=name)
      elif combiner == "sqrtn":
        embeddings = math_ops.segment_sum(embeddings, segment_ids)
        weights_squared = math_ops.pow(weights, 2)
        weight_sum = math_ops.segment_sum(weights_squared, segment_ids)
        weight_sum_sqrt = math_ops.sqrt(weight_sum)
        embeddings = math_ops.div(embeddings, weight_sum_sqrt, name=name)
      else:
        assert False, "Unrecognized combiner"
    else:
      assert idx is not None
      if combiner == "sum":
        embeddings = math_ops.sparse_segment_sum(embeddings, idx, segment_ids,
                                                 name=name)
      elif combiner == "mean":
        embeddings = math_ops.sparse_segment_mean(embeddings, idx, segment_ids,
                                                  name=name)
      elif combiner == "sqrtn":
        embeddings = math_ops.sparse_segment_sqrt_n(embeddings, idx,
                                                    segment_ids, name=name)
      else:
        assert False, "Unrecognized combiner"

    return embeddings

