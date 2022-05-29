import numpy as np
import cv2 as cv
import random, math, sys

IMG_SIZE = 1000
WM_SIZE = 64
BLOCK_SIZE = 8
INCREMENT = 100 # Scaling bigger
EDGE_PIXELS = 50 # Number of pixels to exclude from the edges

THRESHOLD = 127 # Pixels intensity
PIXEL_MAX = 255.0

RANDOM_KEY = 50 # Seed for the pseudorandom generator
FACTOR = 8
ZERO = 0
PROB_SP = 0.05 # Salt and pepper probability

#------------------------FUNCTIONS--------------------------

def psnr(img1, img2): # Peak signal-to-noise ratio
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100    
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def watermark_channels(img, wm):
    channels = np.zeros(img.shape)
    for i in range(3):
        channels[:, :, i] = watermark_image(img[:, :, i], wm)
    cv.imwrite(watermarked_img , channels)
    
    return channels

def watermark_image(img, wm):
	done_x = []
	counter = 0
	img_float = np.float32(img)

	random.seed(RANDOM_KEY) # Initialize the random number generator
	
	img_rows, img_cols = img.shape
	# Exclude the edges:
	c1 = img_rows - EDGE_PIXELS * 2
	c2 = img_cols - EDGE_PIXELS * 2
	
	if(c1 * c2 // (BLOCK_SIZE ** 2) < WM_SIZE ** 2):
		print("The watermark is too large and cannot be embedded.")
		return img

	# Compute the number of blocks:
	total_blocks = (c1 // BLOCK_SIZE) * (c2 // BLOCK_SIZE)
	blocks_needed = WM_SIZE ** 2
	
	# Substitute previous block value with the Inverse Discrete Cosine Transform	
	for i in range(0, img_rows, BLOCK_SIZE):
		for j in range(0, img_cols, BLOCK_SIZE):
			block = img_float[i:i + BLOCK_SIZE, j:j + BLOCK_SIZE]
			dct_block = cv.dct(block / 1.0)		
			block = cv.idct(dct_block)

	while(counter < blocks_needed): # For each pixel of the watermark:
		x = random.randint(1, total_blocks)
		if(x not in done_x):
			done_x.append(x)

			# Change value for 255 or 0
			value = wm[counter // WM_SIZE][counter % WM_SIZE] # Rescaling index
			if(value >= THRESHOLD):
				value = 255
			else:
				value = 0
			wm[counter // WM_SIZE][counter % WM_SIZE] = value
			
			# Computing index:
			m = c1 // BLOCK_SIZE
			n = c2 // BLOCK_SIZE

			i = (x // m) * BLOCK_SIZE + EDGE_PIXELS 
			j = (x % n) * BLOCK_SIZE + EDGE_PIXELS

			# Compute DCT of current block:
			dct_block = cv.dct(img_float[i:i + BLOCK_SIZE, j:j + BLOCK_SIZE] / 1.0)

			# Rounding block value:
			elem = dct_block[ZERO][ZERO] / FACTOR
			elem_ceil = math.ceil(elem)
			if(value == 255): # Round off elem to the nearest ODD number
				if(elem_ceil % 2 == 0): # Even
					elem_ceil -= 1
			else: # value = 0 -> Round off elem to the nearest EVEN number
				if(elem_ceil % 2 == 1): # Odd
					elem_ceil -= 1
			
			dct_block[ZERO][ZERO] = elem_ceil * FACTOR

			# Adding watermark to the pixel:
			img_float[i:i + BLOCK_SIZE, j:j + BLOCK_SIZE] = cv.idct(dct_block)
		
			counter += 1

	# Compute the Peak signal-to-noise ratio:
	print("PSNR is:", psnr(img_float, img))

	return img_float

def extract_channels(img, ext_name):
    channels = np.zeros((WM_SIZE, WM_SIZE, 3))
    print(channels.shape)
    for i in range(3):
        channels[:, :, i] = extract_watermark(img[:, :, i], str(i + 1) + '_' + ext_name)

    return channels

def extract_watermark(img, ext_name):
	done_x = []
	counter = 0
	random.seed(RANDOM_KEY)

	wm = [[0 for x in range(WM_SIZE)] for y in range(WM_SIZE)] # Empty matrix

	if(img.shape != (IMG_SIZE, IMG_SIZE)):
		img = cv.resize(img, (IMG_SIZE, IMG_SIZE))

	c1 = IMG_SIZE - EDGE_PIXELS * 2
	c2 = IMG_SIZE - EDGE_PIXELS * 2

	total_blocks = (c1 // BLOCK_SIZE) * (c2 // BLOCK_SIZE)
	blocks_needed = WM_SIZE ** 2

	while(counter < blocks_needed): # Undo watermark operation
		x = random.randint(1, total_blocks)
		if(x not in done_x):
			done_x.append(x)

			m = c1 // BLOCK_SIZE
			n = c2 // BLOCK_SIZE

			i = (x // m) * BLOCK_SIZE + EDGE_PIXELS
			j = (x % n) * BLOCK_SIZE + EDGE_PIXELS
			
			dct_block = cv.dct(img[i:i + BLOCK_SIZE, j:j + BLOCK_SIZE] / 1.0)
			
			elem = dct_block[ZERO][ZERO]
			elem = math.floor(elem + 0.5) / FACTOR
				
			if(elem % 2 == 0): # Even
				value = 0
			else: # Odd
				value = 255
			
			wm[counter // WM_SIZE][counter % WM_SIZE] = value
			counter += 1
			
	wm = np.array(wm)
		
	cv.imwrite(ext_name , wm)
	print("Watermark extracted and saved in", ext_name)

	return wm

def NCC(img1, img2): # Normalized Cross-Correlation
	num = np.mean((img1 - np.mean(img1)) * (img2 - np.mean(img2)))
	den = np.std(img1) * np.std(img2)
	return abs(num / den)

def ScalingBigger(img):
	bigger = cv.resize(img, (IMG_SIZE + INCREMENT, IMG_SIZE + INCREMENT))
	return bigger

def AverageFilter(img):
	kernel = np.ones((5, 5), np.float32) / 25
	filter = cv.filter2D(img, -1, kernel)
	return filter

def MedianFilter(img):
    m, n = img.shape 
    new_img = np.zeros([m, n])
    for i in range(1, m - 1): 
        for j in range(1, n - 1):
            temp = [img[i - 1, j - 1],
                    img[i - 1, j],
                    img[i - 1, j + 1],
                    img[i, j - 1],
                    img[i, j],
                    img[i, j + 1],
                    img[i + 1, j - 1],
                    img[i + 1, j],
                    img[i + 1, j + 1]]

            temp = sorted(temp) 
            new_img[i, j]= temp[4] 
            new_img = new_img.astype(np.uint8) 
    return new_img

def salt_pepper(image):
	output = np.zeros(image.shape, np.uint8)
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			rdn = random.random()
			if rdn < PROB_SP:
				output[i][j] = 0
			elif rdn > 1 - PROB_SP :
				output[i][j] = 255
			else:
				output[i][j] = image[i][j]
	return output

#-------------------------MAIN CODE----------------------------
if __name__ == "__main__":
	watermarked_img = "Watermarked_Image.jpg"
	watermarked_extracted = "watermarked_extracted.jpg"

	# Reading images received as parameters:
	img_name = sys.argv[1]
	wm_name = sys.argv[2]
	color = sys.argv[3]
	
	img =  cv.imread(img_name) # Original image:
	img = cv.resize(img, dsize=(IMG_SIZE, IMG_SIZE), interpolation=cv.INTER_CUBIC)
	cv.imshow('Original image', img)

	wm = cv.imread(wm_name) # Watermark:
	wm = cv.cvtColor(wm, cv.COLOR_BGR2GRAY)
	cv.imshow('Watermark', wm)
	
	print("--------------------EMBEDDING WATERMARK--------------------")
	if color == 'gray':
		img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		wm = cv.resize(wm, dsize=(WM_SIZE, WM_SIZE), interpolation=cv.INTER_CUBIC)
		
		img_wm = watermark_image(img, wm)
		final_img = np.uint8(img_wm)

		print("Watermarking done!\n--------------------EXTRACTING WATERMARK--------------------")
		wm_extracted = extract_watermark(img_wm, watermarked_extracted)
	
	if color == 'rgb':
		img_wm = watermark_channels(img, wm)
		final_img = np.uint8(img_wm)

		print("Watermarking done!\n--------------------EXTRACTING WATERMARK--------------------")
		wm_extracted = extract_channels(img_wm, watermarked_extracted)
		img_wm = img_wm[:, :, 1]
		wm_extracted = wm_extracted[:, :, 0]	

	cv.imshow('Watermarked image', final_img)
	cv.waitKey(1000)
	
	# Normalized Cross-Correlation:
	print("NCC between original and extracted watermark:", NCC(wm, wm_extracted))

	print("--------------------CHECKING IMAGE VARIATIONS--------------------")
	print("\nChecking when image is scaled to ", IMG_SIZE + INCREMENT, "x", IMG_SIZE + INCREMENT, ":")
	x = ScalingBigger(img_wm)
	wm_extracted = extract_watermark(x, "Extracted_GeoAtt_Bigger.jpg")
	print("NCC:", NCC(wm, wm_extracted))

	print("\nChecking when Average filter is applied:")
	x = AverageFilter(img_wm)
	wm_extracted = extract_watermark(x, "Extracted_SigAtt_AvgFilter.jpg")
	print("NCC:", NCC(wm, wm_extracted))

	print("\nChecking when Median filter is applied:")
	x = MedianFilter(img_wm)
	wm_extracted = extract_watermark(x, "Extracted_SigAtt_MedFilter.jpg")
	print("NCC:", NCC(wm, wm_extracted))

	print("\nChecking when Salt & Pepper noise is added:")
	x = salt_pepper(img_wm)
	wm_extracted = extract_watermark(x, "Extracted_SigAtt_s&pNoise.jpg")
	print("NCC:", NCC(wm, wm_extracted))
