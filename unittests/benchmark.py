import os
import sys
package_path  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, package_path)

from ccorr import cCorrNormedW, wnanmean, quadfit
from scipy import sparse as sps

import numpy as np
from matplotlib import pyplot as plt
import cv2

plt.rcParams['image.origin'] = 'lower'

def prep_template(N=400, period=40, padding=20):
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

def cv_cCoeffNormed_init(img, temp, dMax=20):
	img = img.astype(np.float32)
	temp = temp.astype(np.float32)
	temp = temp[dMax:-dMax, dMax:-dMax]
	return img, temp

def cv_cCoeffNormed_run(img_ppd, temp_ppd, mask=None):
	res = cv2.matchTemplate(img_ppd, temp_ppd, 
			  cv2.TM_CCOEFF_NORMED, mask=mask)
	return res

def gen_mask(r, refimg, center=None):
	ret = np.ones(refimg.shape, dtype=bool)
	xaxis = np.arange(refimg.shape[1])
	yaxis = np.arange(refimg.shape[0])
	if center is None:
		center = [d//2 for d in refimg.shape[::-1]]
	center = np.asarray(center)
	xaxis -= center[0]
	yaxis -= center[1]
	X,Y = np.meshgrid(xaxis,yaxis)
	R = np.hypot(X, Y)
	if isinstance(r, float) or isinstance(r, int):
		Rmin, Rmax = (None, r)
	elif hasattr(r, "__len__") and len(r) ==2:
		Rmin, Rmax = r
	else:
		raise ValueError(f"Unrecognized type r = {r}")
	if Rmin is not None:
		ret = ret & (R>Rmin)
	if Rmax is not None:
		ret = ret & (R<Rmax)
	return ret.astype(refimg.dtype)

def custom_cCoeffNormed_init(temp, mask=None, dMax=20):
	if mask is None:
		mask = np.zeros(temp.shape, dtype=np.float32)
		mask[dMax:-dMax, dMax:-dMax] = 1
		mask = sps.coo_matrix(mask)
	elif isinstance(mask, np.ndarray):
		mask = sps.coo_matrix(mask)
	temp_vec = temp[mask.row, mask.col]
	temp_vec-= wnanmean(temp_vec, mask.data)
	temp_var = wnanmean(temp_vec**2, mask.data)
	# dtemp_coo = sps.coo_matrix((temp_vec, (mask.row, mask.col)))
	return temp_vec, temp_var, mask

def custom_cCoeffNormed_run(img, dtemp_vec, 
	                        temp_var, mask:sps.coo_matrix, dMax=20):
	
	ccoeffmap = cCorrNormedW(img, dtemp_vec, temp_var,
		                     mask.col, mask.row, mask.data, dMax)
	return ccoeffmap

def compare_result_ccoeffmaps(m1, m2, title=None):
	def cm2extent(cm): 
		d = cm.shape
		return (-(d[1]//2)-0.5, d[1]-(d[1]//2)-0.5, 
			    -(d[0]//2)-0.5, d[0]-(d[0]//2)-0.5)
	f1 = plt.figure(figsize=(6,2.5))
	plt.subplot(121)
	plt.imshow(m1, extent=cm2extent(m1))
	plt.colorbar()
	fit_res = quadfit(m1, rk=2)
	k = fit_res["k*"]
	plt.plot(k[0], k[1], 'r+')
	plt.title("CCOEFF map, max@(%.2f,%.2f)"%tuple(k))
	plt.subplot(122)
	dmap = m1-m2
	cmax = abs(dmap).max()
	plt.imshow(dmap, extent=cm2extent(dmap), 
		       cmap='bwr', vmin=-cmax, vmax=cmax)
	plt.colorbar()
	plt.title("Diff between methods")
	if title is not None:
		f1.suptitle(title)
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	from time import perf_counter as pct

	disp = [2,1]
	temp, im = prep_data(200,disp, False)

	## Test 1
	dMax = 20
	print(f"Test #1: {im.shape} image, dMax={dMax}. Mask-free")
	tic = pct()
	img_ppd, temp_ppd = cv_cCoeffNormed_init(im, temp, dMax=dMax)
	toc0 = pct()
	res_cv = cv_cCoeffNormed_run(img_ppd, temp_ppd)
	toc1 = pct()
	print(f"\tcv2.matchTemplate, {temp_ppd.shape} template.")
	print(f"\tInit. time = {(toc0-tic)*1e3:1.2g}ms; Total time = {((toc1-tic)*1e3):1.2g} ms")
	tic = pct()
	dtemp_vec, temp_var, mask_coo = custom_cCoeffNormed_init(temp, dMax=dMax)
	toc0 = pct()
	custom_cCoeffNormed_run(im, dtemp_vec, temp_var, mask_coo, dMax=dMax)
	toc1 = pct()
	print(f"\tccorr.cCorrNormedW")
	print(f"\tInit. time = {(toc0-tic)*1e3:1.2g}ms; Total time = {((toc1-tic)*1e3):1.2g} ms")

	## Test 2
	mask_r = 100#(10,100)
	mask = gen_mask(mask_r, temp_ppd)
	print(f"Test #2: {im.shape} image, dMax={dMax}. Disk Mask r={mask_r}")
	tic = pct()
	img_ppd, temp_ppd = cv_cCoeffNormed_init(im, temp, dMax=dMax)
	toc0 = pct()
	res_cv = cv_cCoeffNormed_run(img_ppd, temp_ppd, mask=mask)
	toc1 = pct()
	print(f"\tUsing cv2.matchTemplate, {temp_ppd.shape} template.",end=' ')
	print(f"Init. time = {(toc0-tic)*1e3:1.2g}ms; Total time = {((toc1-tic)*1e3):1.2g} ms")
	mask = gen_mask(mask_r, temp)
	tic = pct()
	dtemp_vec, temp_var, mask_coo = custom_cCoeffNormed_init(temp, mask=mask, dMax=dMax)
	toc0 = pct()
	res_custom = custom_cCoeffNormed_run(im, dtemp_vec, temp_var, mask_coo, dMax=dMax)
	toc1 = pct()
	print(f"\tccorr.cCorrNormedW")
	print(f"\tInit. time = {(toc0-tic)*1e3:1.2g}ms; Total time = {((toc1-tic)*1e3):1.2g} ms")

	compare_result_ccoeffmaps(res_custom, res_cv, "Test#2")
	

