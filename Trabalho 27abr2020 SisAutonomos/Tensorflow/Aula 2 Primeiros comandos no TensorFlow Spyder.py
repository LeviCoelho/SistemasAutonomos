import tensorflow as tf

frase = tf.constant("Hello World!")
#with tf.Session() as sess:
#   rodar = sess.run(frase)
#print(rodar)
rodar = tf.Session.run(frase) 
print(rodar)