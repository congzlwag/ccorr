# ccorr
Cross-correlator with mask and robust to NAN

The core function is `ccorr.cCorrNormedW`, a variant of [cv2.TM_CCOEFF_NORMED](https://docs.opencv.org/4.x/df/dfb/group__imgproc__object.html#gga3a7850640f1fe1f58fe91a2d7583695dac6677e2af5e0fae82cc5339bfaef5038).
Here is an overview of the comparison
| | `ccorr.cCorrNormedW`    | `cv2.matchTemplate` |
| ---- | ------------------ | ------------------- |
| If there's NAN in the image | Ignores NAN | Breaks |
| Implementation | Overlap-add  | (maybe?) DFT   |
| Range of interest  | Controlled with parameter `dMax` | Prescribed by the shapes of image and template|
| dtype | float64 | float32 |
| Weight in inner-product | $M^2$ | $M$ |

Sometimes there are NAN values in the image and/or the template, sometimes we assign NAN to the locales of imaging artifacts to avoid underestimating the displacement of the pattern of interest. 
`ccorr.cCorrNormedW` is robust to the NAN pixels by ignoring them in any averaging:
$$\mu(f(x',y'), M) \equiv \sum_{(x',y')\in S} f(x',y')M(x',y')\mathbb{1}[f(x',y')\neq \mathrm{NAN}] / \sum_{(x',y')\in S} M(x',y')\mathbb{1}[f(x',y')\neq \mathrm{NAN}]$$
where $M(x',y')$ is the weight mask, sharing the same shape as the template, and $S$ is the support of $M$. 
From the definition of weighted average above, it is also clear that ignoring NAN is different from assigning NAN as 0.
NAN pixels in the template can be simply dropped by the weight mask, but in order to handle NAN in the image, `ccorr.cCorrNormedW` has to abandon FFT and do the convolution in the overlap-add way. 
$$R(x,y) = \frac{\mu(\Delta T(x',y') \Delta I(x+x',y+y'), M)}{\sqrt{\mu(\Delta T(x',y') ^2, M)\mu(\Delta I(x+x',y+y') ^2, M)}}~,$$
which also allows us to specify the $(x,y)$ of interest. 
In contrast, `cv2.matchTemplate` computes at all valid $(x,y)$. 

Indeed if there's no NAN pixel at all, `cv2.matchTemplate` is one order of magnitude faster than `ccorr.cCorrNormedW` at the scale of 200x200 template for 40x40 displacement, but `ccorr.cCorrNormedW` is more accurate when there are imaging artifacts. 
`benchmark.py` compares the performance of `ccorr.cCorrNormedW` to `cv2.matchTemplate` quantitatively.

## Build
    python setup.py build_ext --inplace

The underlying implementation invokes openmp to multi-thread in Cython. 
See [this](https://cython.readthedocs.io/en/latest/src/userguide/parallelism.html) for general practices of parallelization in cython. 

## Functions
In addition, there're `ccorr.quadfit` and `ccorr.calc_ellipse_scale2` to further analyze the maximum and the confidence ellipsoid based on the cross-correlation coefficient map.
Detailed docstring is written in the functions, e.g. `print(ccorr.cCorrNormedW.__doc__)` .
