## Step
1. demo_export.py
3. libtorch

## prepare image
 
    convert frame0000.jpg -crop 1066x800+300+300 frame0000_crop.jpg
    convert 39769.jpg -resize 1066x800 fill_39769.jpg

## Reference
https://github.com/facebookresearch/detr/issues/238

## nsys profile

    nsys profile -o pipeline1.qdstrm --force-overwrite=true --duration=10 --delay=35 ./pipeline1     