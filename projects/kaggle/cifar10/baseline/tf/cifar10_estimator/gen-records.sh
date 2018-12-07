# notice must be python2 for python3 refer to https://www.cs.toronto.edu/~kriz/cifar.html  dict = pickle.load(fo, encoding='bytes')
python=/home/gezi/pyenv/bin/python
$python generate_cifar10_tfrecords.py --data-dir /home/gezi/data/cifar10_data2/
