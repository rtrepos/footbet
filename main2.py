import model2 as mod2
import download as dl
import training_data2 as td2

curr = td2.process_data(2021, "F1")

curr['x_data'].shape

curr['ranking_state']
tdata = td2.get_training_data()

conf = mod2.train_model(10)
conf
print(conf)

mm = mod2.get_model()[0]
mm.summary()

mod2.model_predict()

for k in range(4):
    for i in range(10):
        conf = mod2.train_model(5, False)
        print(conf)
        print(mod2.get_model()[1])
        mod2.model_predict()

    for i in range(10):
        conf = mod2.train_model(5, False)
        print(conf)
        print(mod2.get_model()[1])
        mod2.model_predict()


#mod2.model_predict()

