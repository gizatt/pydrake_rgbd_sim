avconv -framerate 30 -y -i images/%05d_input_depth.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p input_depth.mp4
avconv -framerate 30 -y -i images/%05d_masked_depth.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p masked_depth.mp4
avconv -framerate 30 -y -i images/%05d_mask.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p mask.mp4
rm -fr /tmp/image_tmp
mkdir /tmp/image_tmp
for filename in images/*_rgb.png; do
	name=${filename##*/}
	base=${name%_rgb.png}
	echo $filename
	composite -blend 30 images/${base}_rgb.png images/${base}_masked_depth.png /tmp/image_tmp/${base}_rgb_plus_masked_depth.png
	composite -blend 30 images/${base}_rgb.png images/${base}_mask.png /tmp/image_tmp/${base}_rgb_plus_mask.png
	composite -blend 30 images/${base}_rgb.png images/${base}_input_depth.png /tmp/image_tmp/${base}_rgb_plus_input_depth.png
done
avconv -framerate 30 -y -i /tmp/image_tmp/%05d_rgb_plus_masked_depth.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p rgb_plus_masked_depth.mp4
avconv -framerate 30 -y -i /tmp/image_tmp/%05d_rgb_plus_mask.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p rgb_plus_mask.mp4
avconv -framerate 30 -y -i /tmp/image_tmp/%05d_rgb_plus_input_depth.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p rgb_plus_input_depth.mp4
