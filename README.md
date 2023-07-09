# ccorr
Cross-correlator with mask and robust to NAN

This is a variant of [cv2.TM_CCOEFF_NORMED]([https://docs.opencv.org/4.x/df/dfb/group__imgproc__object.html#gga3a7850640f1fe1f58fe91a2d7583695dac6677e2af5e0fae82cc5339bfaef5038](https://docs.opencv.org/4.x/df/dfb/group__imgproc__object.html#ga586ebfb0a7fb604b35a23d85391329be)).
Sometimes there are NAN values in the image and/or the template, 
and this cross-correlator ignores the NAN values in any averaging in the calculation of the cross correlation coefficient.
In order to handle NAN, it has to abandon FFT and do the convolution in the image space, 
which slows it down (see the output from `benchmark.py`) but makes the displacement detection more accurate.
