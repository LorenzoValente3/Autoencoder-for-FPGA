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
    result_t layer17_out[N_LAYER_16],
    unsigned short &const_size_in_1,
    unsigned short &const_size_out_1
) {

    //hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=encoder_input complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer17_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=encoder_input,layer17_out 
    #pragma HLS PIPELINE 

    const_size_in_1 = N_INPUT_1_1;
    const_size_out_1 = N_LAYER_16;

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        //hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<model_default_t, 3200>(w2, "w2.txt");
        nnet::load_weights_from_txt<model_default_t, 32>(b2, "b2.txt");
        nnet::load_weights_from_txt<model_default_t, 512>(w4, "w4.txt");
        nnet::load_weights_from_txt<model_default_t, 16>(b4, "b4.txt");
        nnet::load_weights_from_txt<model_default_t, 128>(w6, "w6.txt");
        nnet::load_weights_from_txt<model_default_t, 8>(b6, "b6.txt");
        nnet::load_weights_from_txt<model_default_t, 16>(w8, "w8.txt");
        nnet::load_weights_from_txt<model_default_t, 2>(b8, "b8.txt");
        nnet::load_weights_from_txt<model_default_t, 16>(w10, "w10.txt");
        nnet::load_weights_from_txt<model_default_t, 8>(b10, "b10.txt");
        nnet::load_weights_from_txt<model_default_t, 128>(w12, "w12.txt");
        nnet::load_weights_from_txt<model_default_t, 16>(b12, "b12.txt");
        nnet::load_weights_from_txt<model_default_t, 512>(w14, "w14.txt");
        nnet::load_weights_from_txt<model_default_t, 32>(b14, "b14.txt");
        nnet::load_weights_from_txt<model_default_t, 3200>(w16, "w16.txt");
        nnet::load_weights_from_txt<model_default_t, 100>(b16, "b16.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    layer2_t layer2_out[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::dense<input_t, layer2_t, config2>(encoder_input, layer2_out, w2, b2); // dense_45

    layer3_t layer3_out[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    nnet::relu<layer2_t, layer3_t, relu_config3>(layer2_out, layer3_out); // dense_45_relu

    layer4_t layer4_out[N_LAYER_4];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    nnet::dense<layer3_t, layer4_t, config4>(layer3_out, layer4_out, w4, b4); // dense_46

    layer5_t layer5_out[N_LAYER_4];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::relu<layer4_t, layer5_t, relu_config5>(layer4_out, layer5_out); // dense_46_relu

    layer6_t layer6_out[N_LAYER_6];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
    nnet::dense<layer5_t, layer6_t, config6>(layer5_out, layer6_out, w6, b6); // dense_47

    layer7_t layer7_out[N_LAYER_6];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::relu<layer6_t, layer7_t, relu_config7>(layer6_out, layer7_out); // dense_47_relu

    layer8_t layer8_out[N_LAYER_8];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
    nnet::dense<layer7_t, layer8_t, config8>(layer7_out, layer8_out, w8, b8); // encoder_output

    layer9_t layer9_out[N_LAYER_8];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0
    nnet::relu<layer8_t, layer9_t, relu_config9>(layer8_out, layer9_out); // encoder_output_relu

    layer10_t layer10_out[N_LAYER_10];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0
    nnet::dense<layer9_t, layer10_t, config10>(layer9_out, layer10_out, w10, b10); // dense_48

    layer11_t layer11_out[N_LAYER_10];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0
    nnet::relu<layer10_t, layer11_t, relu_config11>(layer10_out, layer11_out); // dense_48_relu

    layer12_t layer12_out[N_LAYER_12];
    #pragma HLS ARRAY_PARTITION variable=layer12_out complete dim=0
    nnet::dense<layer11_t, layer12_t, config12>(layer11_out, layer12_out, w12, b12); // dense_49

    layer13_t layer13_out[N_LAYER_12];
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0
    nnet::relu<layer12_t, layer13_t, relu_config13>(layer12_out, layer13_out); // dense_49_relu

    layer14_t layer14_out[N_LAYER_14];
    #pragma HLS ARRAY_PARTITION variable=layer14_out complete dim=0
    nnet::dense<layer13_t, layer14_t, config14>(layer13_out, layer14_out, w14, b14); // dense_50

    layer15_t layer15_out[N_LAYER_14];
    #pragma HLS ARRAY_PARTITION variable=layer15_out complete dim=0
    nnet::relu<layer14_t, layer15_t, relu_config15>(layer14_out, layer15_out); // dense_50_relu

    layer16_t layer16_out[N_LAYER_16];
    #pragma HLS ARRAY_PARTITION variable=layer16_out complete dim=0
    nnet::dense<layer15_t, layer16_t, config16>(layer15_out, layer16_out, w16, b16); // ecoder_output

    nnet::sigmoid<layer16_t, result_t, sigmoid_config17>(layer16_out, layer17_out); // ecoder_output_sigmoid

}
