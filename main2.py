import model2 as mod2
import download as dl
import training_data2 as td2

#scores = dl.get_all_data()
tdata = td2.get_training_data()

conf = mod2.train_model(0)
conf

mm = mod2.get_model()[0]
mm.summary()

mod2.model_predict()

for i in range(200):
    conf = mod2.train_model(50)
    print(conf)
    print(mod2.get_model()[1])
    mod2.model_predict()

#mod2.model_predict()

