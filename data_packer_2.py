import numpy as np
import os

np.random.seed(5112017)
data_path = "/home/antreas/mlpractical_2016-2017/mlpractical/data"
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

train_data = []
train_labels = []

for subdir, dir, files in os.walk(data_path):
    for file in files:
        if not("html" in file) and not("meta" in file) and not(".txt"in file) and ("msd-25" in file):
            filepath = os.path.join(subdir, file)
            print(filepath)
            data_batch = np.load(filepath)
            print(filepath, data_batch.keys())
            if "test" not in file and "var" not in file:
                train_data.extend(data_batch['inputs'])
                train_labels.extend(data_batch['targets'])

x_train = np.array(train_data)
y_train = np.array(train_labels)



ids = np.arange(x_train.shape[0])
np.random.shuffle(ids)

x_train = x_train[ids]
y_train = y_train[ids]

val_start_index = int(0.75 * x_train.shape[0])
test_start_index = int(0.85 * x_train.shape[0])
print(val_start_index)

x_val = x_train[val_start_index:]
y_val = y_train[val_start_index:]

x_test = x_train[test_start_index:]
y_test = y_train[test_start_index:]

x_train = x_train[:val_start_index]
y_train = y_train[:val_start_index]


# train_pack = np.array({"inputs": x_train, "targets": y_train})
# validation_pack = np.array({"inputs": x_val, "targets": y_val})
# testing_pack = np.array({"inputs": x_test, "targets": y_test})

np.savez("data/msd25-train", inputs=x_train, targets=y_train)
np.savez("data/msd25-valid", inputs=x_val, targets=y_val)
np.savez("data/msd25-test", inputs=x_test, targets=y_test)
print(x_train.shape, y_train.shape, x_val.shape)

