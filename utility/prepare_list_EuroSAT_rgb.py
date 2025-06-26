import os, glob
all_list = []
all_label = []

data_root = os.path.expanduser('~/Data/public_data_AI/EuroSAT')

all_classes = sorted(os.listdir(os.path.join(data_root, 'EuroSAT_RGB')))
# on Mac, there is a hidden file
if '.DS_Store' in all_classes:
    all_classes.remove('.DS_Store')
print(all_classes)


f1 = open(os.path.join(data_root, 'EuroSAT_RGB_image_vs_label.txt'), 'w')
f2 = open(os.path.join(data_root, 'EuroSAT_label_list.txt'), 'w')

for i, cls in enumerate(all_classes):
    print(cls, i)
    f2.writelines(cls + ', ' + str(i) + '\n')

    cls_list = sorted(glob.glob(os.path.join(data_root, 'EuroSAT_RGB' , cls, '*.jpg')))
    # print(cls_list)
    cls_list = ["/".join(dd.split('/')[-2:]) for dd in cls_list]

    for cc in cls_list:
        f1.writelines(cc + ' ' + str(i) + '\n')


f1.close()
f2.close()
