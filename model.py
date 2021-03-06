import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import mutex_utils as mu
import download as dl
import training_data as td


def get_model():
    if mu.mutex_process('data/model.pkl'):
        model_y = tfk.Sequential([
            tfk.layers.InputLayer(batch_input_shape=(None, 5, 8), name='in'),
            tfk.layers.Flatten(),
            tfk.layers.Dense(40, activation=tf.nn.softmax, name='dense1'),
            tfk.layers.Dense(30, activation=tf.nn.softmax, name='dense2'),
            tfk.layers.Dense(10, activation=tf.nn.softmax, name='dense3'),
            #tfk.layers.LSTM(units=5, name='lstm'),
            #tfk.layers.Dense(10, activation=tf.nn.softmax, name='dense'),
            tfk.layers.Dense(3, activation=tf.nn.softmax, name='denseOut')
        ])
        model_y.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        model_y.save('data/model.h5')
        model_acc0 = pd.DataFrame(data={'loss_train': [], 'accuracy_train': [], 'nb_train': [], 'loss_test': [],
                                        'accuracy_test': [], 'nb_test': [], 'nb_epochs':[]})
        mu.mutex_save(model_acc0, 'data/model.pkl')
    mod_y = tfk.models.load_model('data/model.h5')
    mod_acc = mu.mutex_load('data/model.pkl')
    return [mod_y, mod_acc]


def train_model(nb_epochs=0):
    model_y = get_model()[0]
    model_acc = get_model()[1]
    train_data = td.get_training_data()
    x_train = train_data['x_train']
    y_train = train_data['y_train']
    x_test = train_data['x_test']
    y_test = train_data['y_test']
    if nb_epochs > 0:
        model_y.fit(
            x_train,
            y_train,
            batch_size=24,
            epochs=nb_epochs)
    eval_train = model_y.evaluate(x_train, y_train, verbose=2)
    predict_train = tf.math.argmax(model_y.predict(x_train), 1)
    conf_train = tf.math.confusion_matrix(y_train, predict_train)
    #(conf_train[0, 0] + conf_train[1, 1] + conf_train[2, 2]) / np.sum(conf_train)

    eval_test = model_y.evaluate(x_test, y_test, verbose=2)
    predict_test = tf.math.argmax(model_y.predict(x_test), 1)
    conf_test = tf.math.confusion_matrix(y_test, predict_test)
    #(conf_test[0, 0] + conf_test[1, 1] + conf_test[2, 2]) / np.sum(conf_test)
    model_y.save('data/model.h5')
    nb_epochs_tot = nb_epochs
    if model_acc.shape[0] > 0:
        nb_epochs_tot = nb_epochs_tot + model_acc['nb_epochs'][model_acc.shape[0]-1]
    model_acct = pd.DataFrame(data={'loss_train': [eval_train[0]], 'accuracy_train': [eval_train[1]],
                                    'nb_train': [x_train.shape[0]], 'loss_test': [eval_test[0]],
                                    'accuracy_test': [eval_test[1]], 'nb_test': [x_test.shape[0]],
                                    'nb_epochs': [nb_epochs_tot]})
    model_acc = model_acc.append(model_acct, sort=False, ignore_index=True)
    mu.mutex_update(model_acc, 'data/model.pkl')
    return {'conf_train': conf_train, 'conf_test': conf_test}


def model_predict():
    model_y = get_model()[0]
    train_data = td.process_data(2020, 'F1')
    dict_teams = train_data['teams_state']
    home_teams = ["Rennes", "Bordeaux", "Dijon", "Monaco", "Angers", "Lorient", "Nimes", "Reims", "Lille", "Marseille"]
    away_teams = ["Nice", "Metz", "Paris SG", "Brest", "Lens", "St Etienne", "Nantes", "Montpellier", "Strasbourg", "Lyon"]
    x_predict = np.zeros(shape=(1, 5, 8))
    for i in range(len(home_teams)):
        home_team = home_teams[i]
        away_team = away_teams[i]
        x_predict[0, :, 0] = dict_teams[home_team]["BP"]
        x_predict[0, :, 1] = dict_teams[home_team]["BC"]
        x_predict[0, :, 2] = dict_teams[home_team]["cl"]
        x_predict[0, :, 3] = dict_teams[home_team]["loc"]
        x_predict[0, :, 4] = dict_teams[away_team]["BP"]
        x_predict[0, :, 5] = dict_teams[away_team]["BC"]
        x_predict[0, :, 6] = dict_teams[away_team]["cl"]
        x_predict[0, :, 7] = dict_teams[away_team]["loc"]
        prob_pred = model_y.predict(x_predict)
        prediction = tf.math.argmax(prob_pred, 1)
        toprint = "{:>12}\tvs.\t{:>12}\t:\t{:.0f}\t probs: {:.2f}, {:.2f}, {:.2f}".format(
            home_team, away_team, prediction[0], prob_pred[0][0], prob_pred[0][1], prob_pred[0][2])
        print(toprint)
