import model as mod
import download as dl
import training_data as td

#scores = dl.get_all_data()
tdata = td.get_training_data()


conf = mod.train_model(0)
conf
mm = mod.get_model()[0]
mm.summary()

mod.model_predict()

for i in range(200):
    conf = mod.train_model(50)
    print(conf)
    print(mod.get_model()[1])
    mod.model_predict()

# print(mod.get_model()[1])
mod.model_predict()

