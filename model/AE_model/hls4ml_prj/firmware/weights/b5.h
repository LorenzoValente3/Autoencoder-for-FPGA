//Numpy array shape [16]
//Min -0.250000000000
//Max 0.500000000000
//Number of zeros 6

#ifndef B5_H_
#define B5_H_

#ifndef __SYNTHESIS__
bias5_t b5[16];
#else
bias5_t b5[16] = {0.125, 0.000, 0.000, 0.250, -0.250, -0.250, 0.000, 0.125, -0.125, 0.375, 0.000, -0.125, -0.250, 0.000, 0.000, 0.500};
#endif

#endif
