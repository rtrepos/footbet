import pickle
import os


def mutex_process(filename):
    if os.path.exists(filename):
        return False
    filemutex = filename+".mutex"
    try:
        f = open(filemutex, "x")
        f.close()
        return True
    except FileExistsError:
        return False


def mutex_save(obj, filename):
    filemutex = filename+".mutex"
    if os.path.exists(filename):
        raise Exception("[mutex_save] file '" + filename + "' already exists")
    if not os.path.exists(filemutex):
        raise Exception(str("[mutex_save] file '" + filemutex + "' does not exist"))
    outfile = open(filename, 'wb')
    pickle.dump(obj, outfile)
    outfile.close()
    os.remove(filemutex)


def mutex_update(obj, filename):
    outfile = open(filename, 'wb')
    pickle.dump(obj, outfile)
    outfile.close()


def mutex_load(filename):
    pickle_file = open(filename, 'rb')
    obj = pickle.load(pickle_file)
    pickle_file.close()
    return obj

#mutex_process("/tmp/tt.pkl")
#mutex_save({'tt':[0,2]}, "/tmp/tt.pkl")
#tt = mutex_load("/tmp/tt.pkl")
