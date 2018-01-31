import numpy as np
import os

np.random.seed(5112017)
data_path = "data/cifar-10-batches-py"
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

train_data = []
train_labels = []
test_data = []
test_labels = []

for subdir, dir, files in os.walk(data_path):
    for file in files:
        if not("html" in file) and not("meta" in file) and not(".txt"in file):
            filepath = os.path.join(subdir, file)
            print(filepath)
            data_batch = unpickle(filepath)
            print(filepath, data_batch.keys())
            if "test" not in file:
                train_data.extend(data_batch[b'data'])
                train_labels.extend(data_batch[b'labels'])
            else:
                test_data.extend(data_batch[b'data'])
                test_labels.extend(data_batch[b'labels'])

x_train = np.array(train_data)
y_train = np.array(train_labels)

x_test = np.array(test_data)
y_test = np.array(test_labels)

ids = np.arange(x_train.shape[0])
np.random.shuffle(ids)

x_train = x_train[ids]
y_train = y_train[ids]

val_start_index = int(0.85 * x_train.shape[0])
print(val_start_index)

x_val = x_train[val_start_index:]
y_val = y_train[val_start_index:]

x_train = x_train[:val_start_index]
y_train = y_train[:val_start_index]


# train_pack = np.array({"inputs": x_train, "targets": y_train})
# validation_pack = np.array({"inputs": x_val, "targets": y_val})
# testing_pack = np.array({"inputs": x_test, "targets": y_test})

np.savez("data/cifar10-train", inputs=x_train, targets=y_train)
np.savez("data/cifar10-valid", inputs=x_val, targets=y_val)
np.savez("data/cifar10-test", inputs=x_test, targets=y_test)
print(x_train.shape, y_train.shape, x_val.shape)

