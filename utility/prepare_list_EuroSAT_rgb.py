import os, glob
all_list = []
all_label = []

data_root = os.path.expanduser('~/Data/image_classification/EuroSAT')

all_classes = os.listdir(data_root + 'EuroSAT_RGB')
print(all_classes)

f1 = open(os.path.join(data_root, 'EuroSAT_RGB_image_vs_label.txt'), 'w')
f2 = open(os.path.join(data_root, 'EuroSAT_label_list.txt'), 'w')

for i, cls in enumerate(all_classes):
    print(cls, i)
    f2.writelines(cls + ', ' + str(i) + '\n')

    cls_list = glob.glob(os.path.join(data_root, cls, '*.jpg'))
    cls_list = ["/".join(dd.split('/')[-2:]) for dd in cls_list]

    for cc in cls_list:
        f1.writelines(cc + ' ' + str(i) + '\n')


f1.close()
f2.close()
