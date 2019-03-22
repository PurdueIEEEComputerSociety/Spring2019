
DIMENSION=224

#	Most broken files seem to be around 4k
SIZE_CUTOFF=5000

#	$1 = source directory
#	$2 = target directory
function GetScaled()
{
	for file in $1/*;do
		fileName=${file##*/}
		convert $file -resize ${DIMENSION}x${DIMENSION}\! -type truecolor $2/$fileName
	done
}

#	Removes all files of size < targetSize from target directory
#	$1 = target directory
#	$2 = file size
function CleanDir()
{
	for file in $1/*;do
		size=$(wc -c < $file)
		if (( $size < $2)); then
			rm $file
		fi
	done
}

#	Scale Content
# GetScaled ../ieee_csociety_style_transfer_datasets/building scaledContent

#	Scale Styles
# GetScaled ../ieee_csociety_style_transfer_datasets/oilpaint scaledStyle

#	Remove garbage images
CleanDir scaledStyle $SIZE_CUTOFF
CleanDir scaledContent $SIZE_CUTOFF
