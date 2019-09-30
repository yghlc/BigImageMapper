#!/usr/bin/env bash

## Introduction:  Merge NDVI, NDWI, and one RGB band to a three bands image.

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 30 September, 2019

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

PL_API_KEY=$(cat ~/.planetkey)
echo $PL_API_KEY

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


# When an asset is active the direct link to download is present on the asset object in the "location"
download_url=$(curl -L -H "Authorization: api-key $PL_API_KEY" \
    'https://api.planet.com/data/v1/item-types/REOrthoTile/items/20160707_195147_1057916_RapidEye-1/assets/' \
    | jq .visual.location)

echo $download_url

# download the image
#curl -L -H "Authorization: api-key $PL_API_KEY" $download_url > redding.tiff

# need to remove the double quotes surrounding the URL
url=https://api.planet.com/data/v1/download?token=eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJoUmdQY1hZWitxak5TVlVUYk9TR0h4Z05WM2h3TFRTZ1BScUVRb3Z3MXltT1d4QUV5VEh0aHczVWxWQ2pTOUdxZDhpdGEyc3E5NkcvRGRnRzNiQmFMUT09IiwiaXRlbV90eXBlX2lkIjoiUkVPcnRob1RpbGUiLCJ0b2tlbl90eXBlIjoidHlwZWQtaXRlbSIsImV4cCI6MTU2OTg1MjgxNCwiaXRlbV9pZCI6IjIwMTYwNzA3XzE5NTE0N18xMDU3OTE2X1JhcGlkRXllLTEiLCJhc3NldF90eXBlIjoidmlzdWFsIn0.oYBSW3Lrcm6dW4IZyFe9KHYy5j1OKUNF8FrJO_ySFOWHqdZhuPEjo0k53DYqmB96Uy7LgIr4x2TybzyjsZbGUQ
wget -o download_log.txt --output-document=wget_redding.tiff --no-check-certificate $url