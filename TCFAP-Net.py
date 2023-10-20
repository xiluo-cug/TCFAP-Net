#***********************************************MHCA Block***********************************************#
# ########################### MHCA1 ############################
f_sampled_E0 = self.random_sample(f_encoder_list[-6], inputs['sub_idx'][-1])
f_sampled_E1 = self.random_sample(f_encoder_list[-5], inputs['sub_idx'][-1])
f_sampled_E2 = self.random_sample(f_encoder_list[-4], inputs['sub_idx'][-1])
f_sampled_E3 = self.random_sample(f_encoder_list[-3], inputs['sub_idx'][-1])


mca = tf.concat([f_sampled_E0, f_sampled_E1], axis=-1)
mca = tf.concat([mca, f_sampled_E2], axis=-1)
mca = tf.concat([mca, f_sampled_E3], axis=-1)
mca = tf.concat([mca, f_encoder_list[-2]], axis=-1)

Q = tf.reshape(f_encoder_list[-2], shape=[-1, f_encoder_list[-2].get_shape()[2].value, f_encoder_list[-2].get_shape()[3].value])
KV = tf.reshape(mca, shape=[-1, mca.get_shape()[2].value, mca.get_shape()[3].value])

layer = MultiHeadAttention(num_heads=1,key_dim=1)
f_mca = layer(Q,KV)
f_mca = tf.nn.leaky_relu(tf.layers.batch_normalization(f_mca, -1, 0.99, 1e-16, 1e-6,training=is_training))

f_mca = tf.reshape(f_mca, [tf.shape(f_encoder_list[-2])[0], tf.shape(f_encoder_list[-2])[1], 1, f_encoder_list[-2].get_shape()[3].value])
f_encoder_list[-2] = f_mca+f_encoder_list[-2]
# ########################### MHCA2-5 ############################
for i in range(3,7):
    mca = self.nearest_interpolation(mca, inputs['interp_idx'][-(i-1)])
    B = tf.shape(f_encoder_list[-i])[0]
    N = tf.shape(f_encoder_list[-i])[1]
    K = f_encoder_list[-i].get_shape()[2].value
    D = f_encoder_list[-i].get_shape()[3].value
    Q = tf.reshape(f_encoder_list[-i], shape=[-1, K, D])
    KV = tf.reshape(mca, shape=[-1, mca.get_shape()[2].value, mca.get_shape()[3].value])
    
    layer = MultiHeadAttention(num_heads=1,key_dim=1)
    f_mca = layer(Q,KV)
    f_mca = tf.nn.leaky_relu(tf.layers.batch_normalization(f_mca, -1, 0.99, 1e-16,1e-6, training=is_training))

    f_mca = tf.reshape(f_mca, [B, N, 1, D])
    f_encoder_list[-i] = f_mca+f_encoder_list[-i]
# ###########################Bottom############################