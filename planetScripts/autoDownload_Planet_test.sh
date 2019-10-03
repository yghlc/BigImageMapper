#!/usr/bin/env bash

## Introduction:  Merge NDVI, NDWI, and one RGB band to a three bands image.

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 30 September, 2019

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

#PL_API_KEY=$(cat ~/.planetkey)
#echo $PL_API_KEY

# example of downloading one image at : https://developers.planet.com/planetschool/downloading-imagery/

#ItemType: REOrthoTile
#ItemId: 20160707_195147_1057916_RapidEye-1

# get metadata of this ITEM
#curl -L -H "Authorization: api-key $PL_API_KEY" \
#    'https://api.planet.com/data/v1/item-types/REOrthoTile/items/20160707_195147_1057916_RapidEye-1'

## view its footprint in geojson.io,
#curl -L -H "Authorization: api-key $PL_API_KEY" \
#    'https://api.planet.com/data/v1/item-types/REOrthoTile/items/20160707_195147_1057916_RapidEye-1' \
#    | jq '.geometry'| geojsonio

# view asset types
#curl -L -H "Authorization: api-key $PL_API_KEY" \
#    'https://api.planet.com/data/v1/item-types/REOrthoTile/items/20160707_195147_1057916_RapidEye-1/assets' \
#    | jq 'keys'

# check this assets is available to download
#An important thing to know about the API is that it does not pre-generate Assets
#so they are not always immediately availiable to download.
#You can see that the visual asset for this item has the status "inactive", so we need to activate it
#curl -L -H "Authorization: api-key $PL_API_KEY" \
#    'https://api.planet.com/data/v1/item-types/REOrthoTile/items/20160707_195147_1057916_RapidEye-1/assets/' \
#    | jq .analytic.status


## When an asset is active the direct link to download is present on the asset object in the "location"
#download_url=$(curl -L -H "Authorization: api-key $PL_API_KEY" \
#    'https://api.planet.com/data/v1/item-types/REOrthoTile/items/20160707_195147_1057916_RapidEye-1/assets/' \
#    | jq .visual.location)
#
#echo $download_url
#
## download the image
##curl -L -H "Authorization: api-key $PL_API_KEY" $download_url > redding.tiff
#
## need to remove the double quotes surrounding the URL
#url=https://api.planet.com/data/v1/download?token=eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJoUmdQY1hZWitxak5TVlVUYk9TR0h4Z05WM2h3TFRTZ1BScUVRb3Z3MXltT1d4QUV5VEh0aHczVWxWQ2pTOUdxZDhpdGEyc3E5NkcvRGRnRzNiQmFMUT09IiwiaXRlbV90eXBlX2lkIjoiUkVPcnRob1RpbGUiLCJ0b2tlbl90eXBlIjoidHlwZWQtaXRlbSIsImV4cCI6MTU2OTg1MjgxNCwiaXRlbV9pZCI6IjIwMTYwNzA3XzE5NTE0N18xMDU3OTE2X1JhcGlkRXllLTEiLCJhc3NldF90eXBlIjoidmlzdWFsIn0.oYBSW3Lrcm6dW4IZyFe9KHYy5j1OKUNF8FrJO_ySFOWHqdZhuPEjo0k53DYqmB96Uy7LgIr4x2TybzyjsZbGUQ
#wget -o download_log.txt --output-document=wget_redding.tiff --no-check-certificate $url




######################## searching for imagery  ########################
# the tutorial link: https://developers.planet.com/planetschool/searching-for-imagery/

#PSOrthoTile - Images taken by PlanetScope satellites in the OrthoTile format.
#REOrthoTile - Images taken by RapidEye satellites in the OrthoTile format.

#curl -L -H "Authorization: api-key $PL_API_KEY" \
#    'https://api.planet.com/data/v1/item-types' | jq '.item_types[].id'

# this output many ItemTypes ( classes of imagery, some classes represent different satellites,
# some represent different formats, sometimes it's both)
#"MYD09GQ"
#"PSScene4Band"   #  Planet Scope 4 Band
#"SkySatScene"
#"PSScene3Band"   # Planet Scope 3 Band
#"Sentinel1"
#"REScene"
#"REOrthoTile"
#"Sentinel2L1C"
#"MOD09GA"
#"MYD09GA"
#"SkySatCollect"
#"PSOrthoTile"  # Planet Scope OrthoTile
#"Landsat8L1G"
#"MOD09GQ"

export PL_API_KEY=2bd36504c9c247d6a4b0df8794c8c540

imgID=20180521_033449_1_0f1c
item_type=PSScene4Band

url=https://api.planet.com/data/v1/item-types/PSScene4Band/items/20180521_033449_1_0f1c

# show geometry
#curl -L -H "Authorization: api-key $PL_API_KEY" \
#    ${url} | jq '.geometry' | geojsonio

#curl -L -H "Authorization: api-key $PL_API_KEY" ${url}/assets | jq 'keys'
#output:
#[
#  "analytic_dn",
#  "analytic_dn_xml",
#  "basic_analytic_dn",
#  "basic_analytic_dn_nitf",
#  "basic_analytic_dn_rpc",
#  "basic_analytic_dn_rpc_nitf",
#  "basic_analytic_dn_xml",
#  "basic_analytic_dn_xml_nitf",
#  "basic_udm",
#  "udm"
#]

### another image:
#analytic
#analytic_dn
#analytic_dn_xml
#analytic_sr
#analytic_xml
#basic_analytic
#basic_analytic_dn
#basic_analytic_dn_nitf
#basic_analytic_dn_rpc
#basic_analytic_dn_rpc_nitf
#basic_analytic_dn_xml
#basic_analytic_dn_xml_nitf
#basic_analytic_nitf
#basic_analytic_rpc
#basic_analytic_rpc_nitf
#basic_analytic_xml
#basic_analytic_xml_nitf
#basic_udm
#udm

## show active or not
#curl -L -H "Authorization: api-key $PL_API_KEY" ${url}/assets | jq '.udm.status'

url="https://api.planet.com/data/v1/download?token=eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJwUDNCNU9aYVFKUnN2WGsydmF3UVpLL2ZWci9DZWk0bG82OGJuT2NRR2laZ01EcFBTUnpsSWdHNGlZM2R5YTZWQ2xHdDROeFBka29Kb295a1BvdktPUT09IiwiaXRlbV90eXBlX2lkIjoiUkVPcnRob1RpbGUiLCJ0b2tlbl90eXBlIjoidHlwZWQtaXRlbSIsImV4cCI6MTQ3Mzc1MDczOCwiaXRlbV9pZCI6IjIwMTYwNzA3XzE5NTE0N18xMDU3OTE2X1JhcGlkRXllLTEiLCJhc3NldF90eXBlIjoidmlzdWFsIn0.lhRgqIggvnRoCgUVX3hgaNYDQIdU09wVaImxv3a_vuGjfzC7_OteYeViboeiZYBH2_eMdWT5ZWDz2BZiAWkXlQ"
#curl -L "https://api.planet.com/data/v1/download?token=eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJwUDNCNU9aYVFKUnN2WGsydmF3UVpLL2ZWci9DZWk0bG82OGJuT2NRR2laZ01EcFBTUnpsSWdHNGlZM2R5YTZWQ2xHdDROeFBka29Kb295a1BvdktPUT09IiwiaXRlbV90eXBlX2lkIjoiUkVPcnRob1RpbGUiLCJ0b2tlbl90eXBlIjoidHlwZWQtaXRlbSIsImV4cCI6MTQ3Mzc1MDczOCwiaXRlbV9pZCI6IjIwMTYwNzA3XzE5NTE0N18xMDU3OTE2X1JhcGlkRXllLTEiLCJhc3NldF90eXBlIjoidmlzdWFsIn0.lhRgqIggvnRoCgUVX3hgaNYDQIdU09wVaImxv3a_vuGjfzC7_OteYeViboeiZYBH2_eMdWT5ZWDz2BZiAWkXlQ" > download.tif

wget -o download_log.txt --output-document=wget.tiff --no-check-certificate $url