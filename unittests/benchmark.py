import os
import sys
package_path  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, package_path)

from ccorr import cCorrNormedW

import numpy as np
from matplotlib import pyplot as plt
import cv2

def prep_template(N=400, period=30, padding=20):
	xaxis = np.arange(N)-(N//2)
	out = np.cos(xaxis * 2*np.pi/period)**2
	out *= np.exp(-0.5 * (xaxis/(N/10))**2)
	out = out[:, None] * out[None,:]
	return np.pad(out, padding)

def prep_img(template, disp, gs_noise=0.05):
	disp = np.asarray(disp)
	disp = disp.astype(int)
	# assert disp.max() < padding, 'padding should > displacement'
	# out = np.pad(template, padding)
	out = template.copy()
	out = np.roll(out, disp[1], axis=0)
	out = np.roll(out, disp[0], axis=1)
	out += np.random.randn(*(out.shape)) * gs_noise
	return out

def prep_data(N, disp, plot=True):
	temp = prep_template(N)
	im = prep_img(temp, disp)
	diff = im - temp
	if plot:
		f1 = plt.figure(figsize=(6,2.5))
		plt.subplot(121)
		plt.imshow(temp)
		plt.title("Template")
		plt.colorbar()
		plt.subplot(122)
		plt.imshow(diff, cmap='bwr')
		plt.title("Disp=(%g,%g),+noise"%tuple(disp))
		plt.colorbar()
		plt.tight_layout()
		plt.show()
	return temp, im

def cCorrNormed_cv_init(img, temp, dMax=20):
	img = img.astype(np.float32)
	temp = temp.astype(np.float32)
	temp = temp[dMax:-dMax, dMax:-dMax]
	return img, temp

def cCorrNormed_cv(img_ppd, temp_ppd):
	res = cv2.matchTemplate(img_ppd, temp_ppd, 
		      cv2.TM_CCOEFF_NORMED)
	return res

if __name__ == '__main__':
	import timeit
	disp = [2,1]
	temp, im = prep_data(200,disp, False)
	img_ppd, temp_ppd = cCorrNormed_cv_init(im, temp)
	DTcv= timeit.timeit(lambda: cCorrNormed_cv(img_ppd, temp_ppd), number=10)
	print(DTcv/10)

