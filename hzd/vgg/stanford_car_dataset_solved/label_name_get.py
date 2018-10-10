import scipy.io as sio

load_fn = 'mats/cars_annos.mat'
load_data = sio.loadmat(load_fn)

class_names = load_data['class_names']
output = open("lable_name.txt", 'w')
for i in range(class_names.size):
    output.write(" %d : %s \n" % (i + 1, class_names[0][i][0]))

output.close()
