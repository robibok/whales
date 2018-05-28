for path in $1/*
do
filename=$(basename $path)
convert $path -resize 25% $2/$filename
echo $filename
done
