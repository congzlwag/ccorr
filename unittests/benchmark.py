import os
import sys
package_path  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, package_path)

from ccorr import cCorrNormedW, wnanmean, quadfit
from scipy import sparse as sps

import numpy as np
from matplotlib import pyplot as plt
import cv2
from matplotlib.patches import Ellipse

plt.rcParams['image.origin'] = 'lower'

def prep_template(N=400, period=40, padding=20):
	xaxis = np.arange(N)-(N//2)
	out = np.cos(xaxis * 2*np.pi/period)**2
	out *= np.exp(-0.5 * (xaxis/(N/10))**2)
	out = out[:, None] * out[None,:]
	return np.pad(out, padding)

def prep_img(template, disp, gs_noise=0.02):
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
	if plot:
		f1, axs = display_data(temp, im)
		axs[1].set_title("Img Disp=(%g,%g);+noise"%tuple(disp))
		plt.show(block=False)
	return temp, im

def display_data(temp, im):
	diff = im - temp
	f1, axs = plt.subplots(1,3,sharey=True,
		                   figsize=(8,2.5))
	plt.sca(axs[0])
	plt.imshow(temp)
	plt.title("Template")
	plt.colorbar()
	plt.sca(axs[1])
	plt.imshow(im)
	plt.colorbar()
	plt.sca(axs[2])
	plt.imshow(diff, cmap='bwr')
	plt.title("Diff")
	plt.colorbar()
	plt.tight_layout()
	return f1, axs

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

def custom_cCoeffNormed_init(img, temp, mask=None, 
	                         badpixels=None, dMax=20):
	if mask is None:
		mask = np.zeros(temp.shape, dtype=np.float32)
		mask[dMax:-dMax, dMax:-dMax] = 1
		mask = sps.coo_matrix(mask)
	elif isinstance(mask, np.ndarray):
		mask = sps.coo_matrix(mask)
	temp_ppd = temp.copy()
	img_ppd = img.copy()
	if badpixels is not None:
		for roi in badpixels:
			img_ppd[roi] = np.nan
			temp_ppd[roi] = np.nan
	temp_vec = temp_ppd[mask.row, mask.col]
	temp_vec-= wnanmean(temp_vec, mask.data)
	temp_var = wnanmean(temp_vec**2, mask.data)
	# dtemp_coo = sps.coo_matrix((temp_vec, (mask.row, mask.col)))
	return img_ppd, temp_vec, temp_var, mask

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
	f2, axs = plt.subplots(1,3,sharey=True,
		                  figsize=(8,2.5))
	for ax, m in zip(axs, [m1,m2]):
		plt.sca(ax)
		plt.imshow(m, extent=cm2extent(m))
		plt.colorbar()
		fitout = quadfit(m, rk=2)
		visualize_fitres(fitout, s2=1)
		plt.title("max@(%.2f,%.2f)"%tuple(fitout['k*']))
	plt.sca(axs[-1])
	dmap = m1-m2
	cmax = abs(dmap).max()
	plt.imshow(dmap, extent=cm2extent(dmap), 
			   cmap='bwr', vmin=-cmax, vmax=cmax)
	plt.colorbar()
	plt.title("Diff between methods")
	if title is not None:
		f2.suptitle(title)
	plt.tight_layout()

def visualize_fitres(fitout, ax=None, s2=None):
	if ax is None:
		ax = plt.gca()
	ax.plot(*(fitout["k*"]), 'r+')
	ax.axhline(0, c='grey', lw=0.7)
	ax.axvline(0, c='grey', lw=0.7)
	if s2 is None or s2<=0: return
	cov = -np.linalg.inv(fitout["H"]/fitout["err"])
	if np.diag(cov).min() <= 0 or np.linalg.det(cov) <= 0:
		return
	eigenvals, eigenvecs = np.linalg.eig(cov*s2)
	radius_x = np.sqrt(eigenvals[0])
	radius_y = np.sqrt(eigenvals[1])

	ellipse = Ellipse(xy=fitout["k*"], width=radius_x, 
					  height=radius_y, angle=np.rad2deg(np.arccos(eigenvecs[0, 0])),
					  facecolor="none", edgecolor='k')
	ax.add_patch(ellipse)

def blemish(img, artifacts, badpxs):
	imgb = img.copy()
	for roi, val in artifacts:
		imgb[roi] += val
	for roi in badpxs:
		imgb[roi] *= 0
	return imgb


def report(method_name, config, tic, toc0, toc1):
	print(f"\t{method_name}, {config}:")
	print(f"\tInit. time = {(toc0-tic)*1e3:1.2g}ms; Total time = {((toc1-tic)*1e3):1.2g} ms")

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
	report("cv2.matchTemplate", f"template {temp_ppd.shape}", tic, toc0, toc1)
	tic = pct()
	im_ppd, dtemp_vec, temp_var, mask_coo = custom_cCoeffNormed_init(im, temp, dMax=dMax)
	toc0 = pct()
	custom_cCoeffNormed_run(im_ppd, dtemp_vec, temp_var, mask_coo, dMax=dMax)
	toc1 = pct()
	report("ccorr.cCorrNormedW", "", tic, toc0, toc1)

	## Test 2
	mask_r = 100#(10,100)
	mask = gen_mask(mask_r, temp_ppd)
	print(f"Test #2: {im.shape} image, dMax={dMax}. Disk Mask r={mask_r}")
	tic = pct()
	img_ppd, temp_ppd = cv_cCoeffNormed_init(im, temp, dMax=dMax)
	toc0 = pct()
	res_cv = cv_cCoeffNormed_run(img_ppd, temp_ppd, mask=mask)
	toc1 = pct()
	report("cv2.matchTemplate", f"template {temp_ppd.shape}", tic, toc0, toc1)
	mask_coo = gen_mask(mask_r, temp)
	tic = pct()
	im_ppd, dtemp_vec, temp_var, mask_coo = custom_cCoeffNormed_init(im, temp, mask=mask_coo, dMax=dMax)
	toc0 = pct()
	res_custom = custom_cCoeffNormed_run(im, dtemp_vec, temp_var, mask_coo, dMax=dMax)
	toc1 = pct()
	report("ccorr.cCorrNormedW", "", tic, toc0, toc1)

	compare_result_ccoeffmaps(res_custom, res_cv, "Test#2 CCOEFF maps")
	plt.show(block=False)
	
	## Test 3
	artifacts = [(np.s_[115:125,115:125], 1), ]
	badpxs    = [np.s_[138:143, 118:123], ]
	all_badpxs = badpxs + [p[0] for p in artifacts]
	tempb = blemish(temp, artifacts, badpxs)
	imb = blemish(im, artifacts, badpxs)
	f1, _ = display_data(tempb, imb)
	f1.suptitle("Test#3 Blemished Detector")
	plt.show(block=False)
	print(f"Test #3: Blemish Test #2")
	tic = pct()
	img_ppd, temp_ppd = cv_cCoeffNormed_init(imb, tempb, dMax=dMax)
	toc0 = pct()
	res_cv = cv_cCoeffNormed_run(img_ppd, temp_ppd, mask=mask)
	toc1 = pct()
	report("cv2.matchTemplate", f"template {temp_ppd.shape}", tic, toc0, toc1)
	
	mask_coo = mask_coo.toarray()
	for roi in all_badpxs:
		mask_coo[roi] = 0
	mask_coo = sps.coo_matrix(mask_coo)
	tic = pct()
	imb_ppd, dtemp_vec, temp_var, mask_coo = custom_cCoeffNormed_init(imb, tempb, 
		                            mask=mask_coo, badpixels=all_badpxs, dMax=dMax)
	toc0 = pct()
	res_custom = custom_cCoeffNormed_run(imb_ppd, dtemp_vec, temp_var, mask_coo, dMax=dMax)
	toc1 = pct()
	report("ccorr.cCorrNormedW", "", tic, toc0, toc1)

	compare_result_ccoeffmaps(res_custom, res_cv, "Test#3 CCOEFF maps")
	plt.show(block=True)



