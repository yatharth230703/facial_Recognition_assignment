import tensorflow as tf

model_filename = r'C:\Users\Yatharth\Desktop\desktop1\AI\AIMS-Research-Proj\mars-small128.pb'  # Path to the appearance descriptor model

def print_graph_operations(model_filename):
    with tf.compat.v1.Session() as sess:
        with tf.io.gfile.GFile(model_filename, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
        
        for op in sess.graph.get_operations():
            print(op.name)

print_graph_operations(model_filename)
