import numpy as np
import sys
import kmeans
from PIL import Image

# Fago ML HW # 4

def main():
	#read in images through command line
	OGimg = sys.argv[1]
	kval = int(sys.argv[2])

	#open the image using PIL Image
	t_img = Image.open(OGimg, 'r')

	#convert to a format so you see only the RGB values
	RGBvals = t_img.convert('RGB')

	#grab the width and height
	width, height = t_img.size

	# put RGB vals into an array and then reshape it so you can fit all the pixels w*h and include all 3 attributes (R,G,B)
	pvs = np.array(RGBvals)
	#print("pvs shape:", pvs.shape)
	thematrix = np.reshape(pvs, (width*height, 3)) 
	#print("thematrix shape:", thematrix.shape)
	#initialize min error to be large
	min_err = 1000000000

	# run the k value 15 times to find a satisfactory clustering result
	for i in range(15):
		km = kmeans.kmeans(kval,thematrix)
		returntrain = km.kmeanstrain(thematrix)
		returnfwd, error = km.kmeansfwd(thematrix)
		print("curr error: ", error)
		#if this is the iteration with the lowest error, then use this segmentation to create the new image
		if error < min_err:
			min_err = error
			finalfwd = returnfwd
			finaltrain = returntrain

	#prints lowest error from the previous 15 iterations
	print("lowest error: ", min_err)

	#now based on what you know, create a new image with the updated clusters
	
	#put each pixel in the cluster it needs to be in in the matrix
	for i in range(len(finalfwd)):
		thematrix[i] = finaltrain[int(finalfwd[i])]

	# put updated pixels into the new image
	updated = np.reshape(thematrix, (height, width, 3))
	new_image = Image.fromarray(updated, "RGB")


	# compare old and new in a new window
	# want the window to be big enough to fit both so width*2
	together = Image.new("RGB", (width*2,height))
	# OG image
	together.paste(t_img, (0,0))
	# new image
	together.paste(new_image, (width,0))

	#print both out together!
	together.show(together)


main()