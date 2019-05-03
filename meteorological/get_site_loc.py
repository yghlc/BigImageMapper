
import os,sys
path="tem_data/SURF_CLI_CHN_MUL_DAY-TEM-12001-201205.TXT"
id = []
lat = []
lon = []
with open(path,'r') as rf:
    lines = rf.readlines()
    for line in lines:
        str_list = line.split()
        tmp_id = str_list[0]
        if tmp_id in id:
            continue
        else:
            id.append(tmp_id)
            lat.append(str_list[1])
            lon.append(str_list[2])

for idx in range(len(id)):
    print(id[idx],lat[idx],lon[idx])

# save to csv file
# Latitude,Longitude,Name
# 48.1,0.25,"First point"
# 49.2,1.1,"Second point"
# 47.5,0.75,"Third point"
with open('site_locations.csv','w') as wf:
    wf.writelines("Latitude,Longitude,Name"+'\n')
    for idx in range(len(id)):
        # the last two is Minute, other is degree
        tmp_lat = float(lat[idx][-2:])/60 + float(lat[idx][:-2])
        tmp_lon = float(lon[idx][-2:])/60 + float(lon[idx][:-2])
        wf.writelines("%f, %f, %s\n"%(tmp_lat,tmp_lon,id[idx]))
