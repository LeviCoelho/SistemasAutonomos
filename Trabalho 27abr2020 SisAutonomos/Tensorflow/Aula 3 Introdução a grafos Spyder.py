import tensorflow as tf

a = tf.constant(5)
b = tf.constant(3)
c = tf.constant(2)

d = tf.multiply(a,b)
e = tf.add(b,c)
f = tf.subtract(d,e)

sess = tf.Session()
saida = sess.run(f)
sess.close()