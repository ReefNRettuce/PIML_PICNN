# Make all instances of person 255 pixel value and background 0.
im = mask > 0
mask[im] = 255
mask[np.logical_not(im)] = 0