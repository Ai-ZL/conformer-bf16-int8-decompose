#!/usr/bin/env python3

import kaldi_native_fbank as knf
import numpy as np
from onnxruntime.quantization import QuantType, quantize_static, CalibrationDataReader,QuantFormat, CalibrationMethod
from datasets import load_dataset


def compute_feat(samples,sample_rate):
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = sample_rate
    opts.mel_opts.num_bins = 80

    online_fbank = knf.OnlineFbank(opts)
    online_fbank.accept_waveform(sample_rate, (samples * 32768).tolist())
    online_fbank.input_finished()

    features = np.stack(
        [online_fbank.get_frame(i) for i in range(online_fbank.num_frames_ready)]
    )
    assert features.data.contiguous is True
    assert features.dtype == np.float32, features.dtype
    mean = features.mean(axis=0, keepdims=True)
    stddev = features.std(axis=0, keepdims=True)
    features = (features - mean) / (stddev + 1e-5)
    return features


class ConformerDataReader(CalibrationDataReader):
    def __init__(self):

        self.enum_data_dicts = []
        self.datasize = 0
        librispeech_eval = load_dataset("openslr/librispeech_asr", "clean",
                                        split="test",trust_remote_code=True)  # change to "other" for other test dataset

        Features=[]
        for i in [583, 687, 739, 330, 447, 700, 1046, 1065,1071, 1085, 1328, 1362,1516,  1578,1703, 1731, 1938, 2213,2285, 2600,2605,2606,2607,2609,2610,2611,2612,2613,2615]:#audio sample whose WER larger than 0.5
            features = compute_feat(librispeech_eval[i]['audio']['array'],
                                        librispeech_eval[i]['audio']['sampling_rate'])  # (T, C)
            features = np.expand_dims(features, axis=0)  # (N, T, C)
            features = features.transpose(0, 2, 1)  # (N, C, T)
            if i==583:
                Features=features
            else:
                Features=np.append(Features,features,axis=2)
        print(Features.shape)
        features_length = np.array([features.shape[2]], dtype=np.int64)
        self.enum_data_dicts.append({"audio_signal": features, "length": features_length})
        self.datasize = len(self.enum_data_dicts)
        self.enum_data_dicts = iter(self.enum_data_dicts)

    def get_next(self):
        return next(self.enum_data_dicts, None)


def main():
    dr = ConformerDataReader()
    q_static_opts = {
                     "WeightSymmetric": True,
                     "ActivationSymmetric": True
                     }

    quantize_static(
        model_input="model_opt.onnx", #the original model path
        model_output="model_opt_I8I8_qops_static_symw+a.onnx", #the quantized model
        calibration_data_reader=dr,
        quant_format=QuantFormat.QOperator,
        per_channel=True,
        weight_type=QuantType.QInt8,
        nodes_to_exclude=['/layers.0/feed_forward1/activation/Sigmoid','/layers.0/conv/Sigmoid','/layers.0/conv/activation/Sigmoid',
                          '/layers.0/feed_forward2/activation/Sigmoid','/layers.1/feed_forward1/activation/Sigmoid','/layers.1/conv/Sigmoid','/layers.1/conv/activation/Sigmoid',
                          '/layers.1/feed_forward2/activation/Sigmoid','/layers.2/feed_forward1/activation/Sigmoid','/layers.2/conv/Sigmoid','/layers.2/conv/activation/Sigmoid',
                          '/layers.2/feed_forward2/activation/Sigmoid','/layers.3/feed_forward1/activation/Sigmoid','/layers.3/conv/Sigmoid','/layers.3/conv/activation/Sigmoid',
                          '/layers.3/feed_forward2/activation/Sigmoid','/layers.4/feed_forward1/activation/Sigmoid','/layers.4/conv/Sigmoid','/layers.4/conv/activation/Sigmoid',
                          '/layers.4/feed_forward2/activation/Sigmoid','/layers.5/feed_forward1/activation/Sigmoid','/layers.5/conv/Sigmoid','/layers.5/conv/activation/Sigmoid',
                          '/layers.5/feed_forward2/activation/Sigmoid','/layers.6/feed_forward1/activation/Sigmoid','/layers.6/conv/Sigmoid','/layers.6/conv/activation/Sigmoid',
                          '/layers.6/feed_forward2/activation/Sigmoid','/layers.7/feed_forward1/activation/Sigmoid','/layers.7/conv/Sigmoid','/layers.7/conv/activation/Sigmoid',
                          '/layers.7/feed_forward2/activation/Sigmoid','/layers.8/feed_forward1/activation/Sigmoid','/layers.8/conv/Sigmoid','/layers.8/conv/activation/Sigmoid',
                          '/layers.8/feed_forward2/activation/Sigmoid','/layers.9/feed_forward1/activation/Sigmoid','/layers.9/conv/Sigmoid','/layers.9/conv/activation/Sigmoid',
                          '/layers.9/feed_forward2/activation/Sigmoid','/layers.10/feed_forward1/activation/Sigmoid','/layers.10/conv/Sigmoid','/layers.10/conv/activation/Sigmoid',
                          '/layers.10/feed_forward2/activation/Sigmoid','/layers.11/feed_forward1/activation/Sigmoid','/layers.11/conv/Sigmoid','/layers.11/conv/activation/Sigmoid',
                          '/layers.11/feed_forward2/activation/Sigmoid','/layers.12/feed_forward1/activation/Sigmoid','/layers.12/conv/Sigmoid','/layers.12/conv/activation/Sigmoid',
                          '/layers.12/feed_forward2/activation/Sigmoid','/layers.13/feed_forward1/activation/Sigmoid','/layers.13/conv/Sigmoid','/layers.13/conv/activation/Sigmoid',
                          '/layers.13/feed_forward2/activation/Sigmoid','/layers.14/feed_forward1/activation/Sigmoid','/layers.14/conv/Sigmoid','/layers.14/conv/activation/Sigmoid',
                          '/layers.14/feed_forward2/activation/Sigmoid','/layers.15/feed_forward1/activation/Sigmoid','/layers.15/conv/Sigmoid','/layers.15/conv/activation/Sigmoid',
                          '/layers.15/feed_forward2/activation/Sigmoid'],
        calibrate_method=CalibrationMethod.Entropy,
        extra_options=q_static_opts
    )


if __name__ == "__main__":
    main()
