import tensorflow as tf
import pandas as pd


def getdata(filepath):
    file = open(filepath,'r')
    data = file.readlines()
    if '\n' in data:
        data.remove('\n')
    datas = []
    for item in data:
        datas.append(item.strip().split(','))
    return datas


def train_func(train_x,train_y):
    dataset=tf.data.Dataset.from_tensor_slices((dict(train_x), train_y))
    dataset = dataset.shuffle(1000).repeat().batch(100)
    return dataset


def eval_input_fn(features, labels, batch_size):
    features = dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.batch(batch_size)
    return dataset


if __name__ == "__main__":

    CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
    data_train = pd.read_csv('data.csv', names=CSV_COLUMN_NAMES, header=0)
    data_test = pd.read_csv('test.csv', names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = data_train, data_train.pop('Species')
    test_x, test_y = data_test, data_test.pop('Species')
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    classifier = tf.estimator.DNNClassifier(feature_columns=my_feature_columns, hidden_units=[10, 10], n_classes=3)
    classifier.train(input_fn=lambda: train_func(train_x, train_y),steps=1000)
    predict_arr = []
    predictions = classifier.predict(
        input_fn=lambda: eval_input_fn(test_x, labels=test_y, batch_size=100))
    for predict in predictions:
        predict_arr.append(predict['probabilities'].argmax())
    result = predict_arr == test_y
    result1 = [w for w in result if w == True]
    print("Precision: %s" % str((len(result1) / len(result))))

