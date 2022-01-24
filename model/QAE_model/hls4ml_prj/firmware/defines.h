#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

//hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 64
#define N_LAYER_2 32
#define N_LAYER_5 16
#define N_LAYER_8 2
#define N_LAYER_10 16
#define N_LAYER_12 16
#define N_LAYER_16 32
#define N_LAYER_18 32
#define N_LAYER_22 64
#define N_LAYER_24 10

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> layer2_t;
typedef ap_fixed<4,1> weight2_t;
typedef ap_fixed<4,1> bias2_t;
typedef ap_fixed<16,6,AP_RND,AP_SAT> layer3_t;
typedef ap_ufixed<4,0,AP_RND,AP_SAT> layer4_t;
typedef ap_fixed<16,6> layer5_t;
typedef ap_fixed<4,1> weight5_t;
typedef ap_fixed<4,1> bias5_t;
typedef ap_fixed<16,6,AP_RND,AP_SAT> layer6_t;
typedef ap_ufixed<4,0,AP_RND,AP_SAT> layer7_t;
typedef ap_fixed<16,6> layer8_t;
typedef ap_fixed<16,6> encoder_output_weight_t;
typedef ap_fixed<16,6> encoder_output_bias_t;
typedef ap_fixed<16,6> encoder_output_relu_default_t;
typedef ap_fixed<16,6,AP_RND,AP_SAT> layer9_t;
typedef ap_fixed<16,6> layer10_t;
typedef ap_fixed<5,1> weight10_t;
typedef ap_fixed<5,1> bias10_t;
typedef ap_fixed<16,6,AP_RND,AP_SAT> layer11_t;
typedef ap_fixed<16,6> layer12_t;
typedef ap_fixed<5,1> weight12_t;
typedef ap_fixed<5,1> bias12_t;
typedef ap_fixed<16,6,AP_RND,AP_SAT> layer13_t;
typedef ap_ufixed<5,0,AP_RND,AP_SAT> layer14_t;
typedef ap_ufixed<5,0,AP_RND,AP_SAT> layer15_t;
typedef ap_fixed<16,6> layer16_t;
typedef ap_fixed<5,1> weight16_t;
typedef ap_fixed<5,1> bias16_t;
typedef ap_fixed<16,6,AP_RND,AP_SAT> layer17_t;
typedef ap_fixed<16,6> layer18_t;
typedef ap_fixed<5,1> weight18_t;
typedef ap_fixed<5,1> bias18_t;
typedef ap_fixed<16,6,AP_RND,AP_SAT> layer19_t;
typedef ap_ufixed<5,0,AP_RND,AP_SAT> layer20_t;
typedef ap_ufixed<5,0,AP_RND,AP_SAT> layer21_t;
typedef ap_fixed<16,6> layer22_t;
typedef ap_fixed<5,1> weight22_t;
typedef ap_fixed<5,1> bias22_t;
typedef ap_fixed<16,6,AP_RND,AP_SAT> layer23_t;
typedef ap_fixed<16,6> layer24_t;
typedef ap_fixed<16,7> weight24_t;
typedef ap_fixed<16,7> bias24_t;
typedef ap_fixed<16,6,AP_RND,AP_SAT> layer25_t;
typedef ap_fixed<16,6> sigmoid_default_t;
typedef ap_fixed<16,6> result_t;
typedef ap_fixed<16,6> classifier_output_default_t;

#endif
