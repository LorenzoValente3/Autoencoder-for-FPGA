//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    input_t encoder_input[N_INPUT_1_1],
    result_t layer26_out[N_LAYER_22], result_t layer27_out[N_LAYER_24],
    unsigned short &const_size_in_1,
    unsigned short &const_size_out_1, unsigned short &const_size_out_2
) {

    //hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=encoder_input complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer26_out complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer27_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=encoder_input,layer26_out,layer27_out 
    #pragma HLS PIPELINE 

    const_size_in_1 = N_INPUT_1_1;
    const_size_out_1 = N_LAYER_22;
    const_size_out_2 = N_LAYER_24;

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        //hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight2_t, 2048>(w2, "w2.txt");
        nnet::load_weights_from_txt<bias2_t, 32>(b2, "b2.txt");
        nnet::load_weights_from_txt<weight5_t, 512>(w5, "w5.txt");
        nnet::load_weights_from_txt<bias5_t, 16>(b5, "b5.txt");
        nnet::load_weights_from_txt<encoder_output_weight_t, 32>(w8, "w8.txt");
        nnet::load_weights_from_txt<encoder_output_bias_t, 2>(b8, "b8.txt");
        nnet::load_weights_from_txt<weight10_t, 32>(w10, "w10.txt");
        nnet::load_weights_from_txt<bias10_t, 16>(b10, "b10.txt");
        nnet::load_weights_from_txt<weight12_t, 32>(w12, "w12.txt");
        nnet::load_weights_from_txt<bias12_t, 16>(b12, "b12.txt");
        nnet::load_weights_from_txt<weight16_t, 512>(w16, "w16.txt");
        nnet::load_weights_from_txt<bias16_t, 32>(b16, "b16.txt");
        nnet::load_weights_from_txt<weight18_t, 512>(w18, "w18.txt");
        nnet::load_weights_from_txt<bias18_t, 32>(b18, "b18.txt");
        nnet::load_weights_from_txt<weight22_t, 2048>(w22, "w22.txt");
        nnet::load_weights_from_txt<bias22_t, 64>(b22, "b22.txt");
        nnet::load_weights_from_txt<weight24_t, 320>(w24, "w24.txt");
        nnet::load_weights_from_txt<bias24_t, 10>(b24, "b24.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    layer2_t layer2_out[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::dense<input_t, layer2_t, config2>(encoder_input, layer2_out, w2, b2); // fc1

    layer3_t layer3_out[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    nnet::linear<layer2_t, layer3_t, linear_config3>(layer2_out, layer3_out); // fc1_linear

    layer4_t layer4_out[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    nnet::relu<layer3_t, layer4_t, relu_config4>(layer3_out, layer4_out); // relu1

    layer5_t layer5_out[N_LAYER_5];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::dense<layer4_t, layer5_t, config5>(layer4_out, layer5_out, w5, b5); // fc2_prun

    layer6_t layer6_out[N_LAYER_5];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
    nnet::linear<layer5_t, layer6_t, linear_config6>(layer5_out, layer6_out); // fc2_prun_linear

    layer7_t layer7_out[N_LAYER_5];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::relu<layer6_t, layer7_t, relu_config7>(layer6_out, layer7_out); // relu2

    layer8_t layer8_out[N_LAYER_8];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
    nnet::dense<layer7_t, layer8_t, config8>(layer7_out, layer8_out, w8, b8); // encoder_output

    layer9_t layer9_out[N_LAYER_8];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0
    nnet::relu<layer8_t, layer9_t, relu_config9>(layer8_out, layer9_out); // encoder_output_relu

    layer10_t layer10_out[N_LAYER_10];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0
    nnet::dense<layer9_t, layer10_t, config10>(layer9_out, layer10_out, w10, b10); // fc3

    layer11_t layer11_out[N_LAYER_10];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0
    nnet::linear<layer10_t, layer11_t, linear_config11>(layer10_out, layer11_out); // fc3_linear

    layer12_t layer12_out[N_LAYER_12];
    #pragma HLS ARRAY_PARTITION variable=layer12_out complete dim=0
    nnet::dense<layer9_t, layer12_t, config12>(layer9_out, layer12_out, w12, b12); // fc4_prunedclass

    layer13_t layer13_out[N_LAYER_12];
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0
    nnet::linear<layer12_t, layer13_t, linear_config13>(layer12_out, layer13_out); // fc4_prunedclass_linear

    layer14_t layer14_out[N_LAYER_10];
    #pragma HLS ARRAY_PARTITION variable=layer14_out complete dim=0
    nnet::relu<layer11_t, layer14_t, relu_config14>(layer11_out, layer14_out); // relu3

    layer15_t layer15_out[N_LAYER_12];
    #pragma HLS ARRAY_PARTITION variable=layer15_out complete dim=0
    nnet::relu<layer13_t, layer15_t, relu_config15>(layer13_out, layer15_out); // prunclass_relu4

    layer16_t layer16_out[N_LAYER_16];
    #pragma HLS ARRAY_PARTITION variable=layer16_out complete dim=0
    nnet::dense<layer14_t, layer16_t, config16>(layer14_out, layer16_out, w16, b16); // fc4

    layer17_t layer17_out[N_LAYER_16];
    #pragma HLS ARRAY_PARTITION variable=layer17_out complete dim=0
    nnet::linear<layer16_t, layer17_t, linear_config17>(layer16_out, layer17_out); // fc4_linear

    layer18_t layer18_out[N_LAYER_18];
    #pragma HLS ARRAY_PARTITION variable=layer18_out complete dim=0
    nnet::dense<layer15_t, layer18_t, config18>(layer15_out, layer18_out, w18, b18); // fc5_class

    layer19_t layer19_out[N_LAYER_18];
    #pragma HLS ARRAY_PARTITION variable=layer19_out complete dim=0
    nnet::linear<layer18_t, layer19_t, linear_config19>(layer18_out, layer19_out); // fc5_class_linear

    layer20_t layer20_out[N_LAYER_16];
    #pragma HLS ARRAY_PARTITION variable=layer20_out complete dim=0
    nnet::relu<layer17_t, layer20_t, relu_config20>(layer17_out, layer20_out); // relu4

    layer21_t layer21_out[N_LAYER_18];
    #pragma HLS ARRAY_PARTITION variable=layer21_out complete dim=0
    nnet::relu<layer19_t, layer21_t, relu_config21>(layer19_out, layer21_out); // class_relu5

    layer22_t layer22_out[N_LAYER_22];
    #pragma HLS ARRAY_PARTITION variable=layer22_out complete dim=0
    nnet::dense<layer20_t, layer22_t, config22>(layer20_out, layer22_out, w22, b22); // decoder_output

    layer23_t layer23_out[N_LAYER_22];
    #pragma HLS ARRAY_PARTITION variable=layer23_out complete dim=0
    nnet::linear<layer22_t, layer23_t, linear_config23>(layer22_out, layer23_out); // decoder_output_linear

    layer24_t layer24_out[N_LAYER_24];
    #pragma HLS ARRAY_PARTITION variable=layer24_out complete dim=0
    nnet::dense<layer21_t, layer24_t, config24>(layer21_out, layer24_out, w24, b24); // classifier_out

    layer25_t layer25_out[N_LAYER_24];
    #pragma HLS ARRAY_PARTITION variable=layer25_out complete dim=0
    nnet::linear<layer24_t, layer25_t, linear_config25>(layer24_out, layer25_out); // classifier_out_linear

    nnet::sigmoid<layer23_t, result_t, sigmoid_config26>(layer23_out, layer26_out); // sigmoid

    nnet::softmax<layer25_t, result_t, softmax_config27>(layer25_out, layer27_out); // classifier_output

}
