
#
# class InstantLayerNormalization(Layer):
#     '''
#     Class implementing instant layer normalization. It can also be called
#     channel-wise layer normalization and was proposed by
#     Luo & Mesgarani (https://arxiv.org/abs/1809.07454v2)
#     '''
#
#     def __init__(self, **kwargs):
#         '''
#             Constructor
#         '''
#         super(InstantLayerNormalization, self).__init__(**kwargs)
#         self.epsilon = 1e-7
#         self.gamma = None
#         self.beta = None
#
#     def build(self, input_shape):
#         '''
#         Method to build the weights.
#         '''
#         shape = input_shape[-1:]
#         # initialize gamma
#         self.gamma = self.add_weight(shape=shape,
#                                      initializer='ones',
#                                      trainable=True,
#                                      name='gamma')
#         # initialize beta
#         self.beta = self.add_weight(shape=shape,
#                                     initializer='zeros',
#                                     trainable=True,
#                                     name='beta')
#
#     def call(self, inputs):
#         '''
#         Method to call the Layer. All processing is done here.
#         '''
#
#
#         # calculate mean of each frame
#         mean = tf.math.reduce_mean(inputs, axis=[-1], keepdims=True)
#         # calculate variance of each frame
#         variance = tf.math.reduce_mean(tf.math.square(inputs - mean),
#                                        axis=[-1], keepdims=True)
#         # calculate standard deviation
#         std = tf.math.sqrt(variance + self.epsilon)
#         # normalize each frame independently
#         outputs = (inputs - mean) / std
#         # scale with gamma
#         outputs = outputs * self.gamma
#         # add the bias beta
#         outputs = outputs + self.beta
#         # return output
#         return outputs