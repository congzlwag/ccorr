# ccorr
Cross-correlator with mask and robust to NAN

This is a variant of [cv2.TM_CCOEFF_NORMED](https://docs.opencv.org/4.x/df/dfb/group__imgproc__object.html#gga3a7850640f1fe1f58fe91a2d7583695dac6677e2af5e0fae82cc5339bfaef5038).
Sometimes there are NAN values in the image and/or the template, 
and this cross-correlator ignores the NAN values in any averaging in the calculation of the cross correlation coefficient.
In order to handle NAN, it has to abandon FFT and do the convolution in the image space.
Compare to `cv2.matchTemplate`, here are the pros and cons
| Pros                        | Cons |
| ---- | ---- |
| More accurate with proper   | Slower |
| small displacement  | a |

see the output from `benchmark.py`

The underlying implementation invokes openmp to multi-thread in cython. See [this](https://cython.readthedocs.io/en/latest/src/userguide/parallelism.html) for general practices of parallelization in cython.

## FAQ
### Why are there NAN in the first place?
Aside from readout errors, it is common that there are bad pixels or artifacts that are fixed in the image space. These structures can cheat a plain cross-correlator to bias the displacement estimation towards 0.
