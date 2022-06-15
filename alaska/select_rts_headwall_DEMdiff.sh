#!/bin/bash

#proc=20
#proc=2
proc=1
# for selecting headwall lines, must set a buffer size
buffer=50

py=~/codes/PycharmProjects/Landuse_DL/alaska/select_rts_from_YOLO_demDiff_headwall.py

# selection grid by gird
#${py} ${ext_shp} -g --process_num=${proc}

# selction by input two shapefiles
headwall_lines=dem_headwall_shp_grid/headwall_shps_grid10741/headwall_shp_multiDates_10741_subset_rippleSel.shp
dem_diff=grid_dem_diffs_segment_results/segment_result_grid10741/grid_ids_DEM_diff_grid10741_8bit_post.shp

${py} ${headwall_lines}  ${dem_diff} --process_num=${proc} --buffer_size=${buffer}

