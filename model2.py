import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import mutex_utils as mu
import download as dl
import training_data2 as td2


def get_model():
    if mu.mutex_process('data/model2.pkl'):
        model_y = tfk.Sequential([
            tfk.layers.InputLayer(batch_input_shape=(None, 20, 8+6+15), name='in'),
            tfk.layers.Conv1D(filters=100, kernel_size=2, strides=1, name='conv1'),
            tfk.layers.MaxPooling1D(pool_size=4, name='pool1'),
            tfk.layers.Dropout(.2),
            tfk.layers.Conv1D(filters=50, kernel_size=3, strides=1, name='conv2'),
            tfk.layers.MaxPooling1D(pool_size=2, name='pool2'),
            tfk.layers.Dropout(.2),
            tfk.layers.Flatten(name='flat'),
            tfk.layers.Dense(50, activation=tfk.activations.linear, name='dense1'),
            tfk.layers.Dropout(.2),
            tfk.layers.Dense(20, activation=tfk.activations.linear, name='dense2'),
            tfk.layers.Dropout(.2),
            tfk.layers.Dense(3, activation=tf.nn.softmax, name='denseOut')
        ])
        model_y.summary()
        model_y.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        model_y.save('data/model2.h5')
        model_acc0 = pd.DataFrame(data={'loss_train': [], 'accuracy_train': [], 'nb_train': [], 'loss_test': [],
                                        'accuracy_test': [], 'nb_test': [], 'nb_epochs':[]})
        mu.mutex_save(model_acc0, 'data/model2.pkl')
    mod_y = tfk.models.load_model('data/model2.h5')
    mod_acc = mu.mutex_load('data/model2.pkl')
    return [mod_y, mod_acc]


def train_model(nb_epochs=0, on_test=False):
    model_y = get_model()[0]
    model_acc = get_model()[1]
    train_data = td2.get_training_data()
    x_train = train_data['x_train']
    y_train = train_data['y_train']
    x_test = train_data['x_test']
    y_test = train_data['y_test']
    if nb_epochs > 0 and not on_test:
        model_y.fit(
            x_train,
            y_train,
            batch_size=24,
            epochs=nb_epochs)
    if nb_epochs > 0 and on_test:
        model_y.fit(
            x_test,
            y_test,
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
    model_y.save('data/model2.h5')
    nb_epochs_tot = nb_epochs
    if model_acc.shape[0] > 0:
        nb_epochs_tot = nb_epochs_tot + model_acc['nb_epochs'][model_acc.shape[0]-1]
    nb_train = x_train.shape[0]
    if on_test:
        nb_train = x_test.shape[0]
    model_acct = pd.DataFrame(data={'loss_train': [eval_train[0]], 'accuracy_train': [eval_train[1]],
                                    'nb_train': [x_train.shape[0]], 'loss_test': [eval_test[0]],
                                    'accuracy_test': [eval_test[1]], 'nb_test': [x_test.shape[0]],
                                    'nb_epochs': [nb_epochs_tot], 'tain_on_test': [on_test]})
    model_acc = model_acc.append(model_acct, sort=False, ignore_index=True)
    mu.mutex_update(model_acc, 'data/model2.pkl')
    return {'conf_train': conf_train, 'conf_test': conf_test}


def model_predict():
    model_y = get_model()[0]
    train_data = td2.process_data(2021, 'F1')
    classement = train_data['ranking_state']
    matches = np.array([[ "St Etienne",  "Angers"],
                        [ "Nantes", "Clermont"],
                        [ "Lille", "Brest"],
                        [ "Nice", "Lyon"],
                        [ "Lens", "Metz"],
                        [ "Lorient", "Bordeaux"],
                        [ "Reims", "Troyes" ],
                        [ "Rennes", "Strasbourg"],
                        [ "Monaco", "Montpellier"],
                        [ "Marseille", "Paris SG"]])
    home_teams = matches[:, 0]
    away_teams = matches[:, 1]
    x_predict = np.zeros(shape=(1, 20, 8+6+15))
    for i in range(len(home_teams)):
        hometeam = home_teams[i]
        awayteam = away_teams[i]
        cl = pd.concat([classement.loc[classement.team == hometeam, :],
                        classement.loc[classement.team == awayteam, :],
                        classement.loc[np.logical_and(classement.team != hometeam, classement.team != awayteam), :]])
        x_predict[0, :, :] = cl.drop('team', axis=1)
        prob_pred = model_y.predict(x_predict)
        prediction = tf.math.argmax(prob_pred, 1)
        toprint = "{:>12}\tvs.\t{:>12}\t:\t{:.0f}\t probs: {:.2f}, {:.2f}, {:.2f}".format(
            hometeam, awayteam, prediction[0], prob_pred[0][0], prob_pred[0][1], prob_pred[0][2])
        print(toprint)
