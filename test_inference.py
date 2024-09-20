import kaldi_native_fbank as knf
import itertools
import librosa
import numpy as np
import onnxruntime as ort
import time
import argparse, os


parser = argparse.ArgumentParser()
parser.add_argument('-model', '--model_path', help='model stored path')
parser.add_argument('-data', '--data_path', help='wav test data stored path')
parser.add_argument('-t', '--tokens_path', help='tokens list stored path')

def compute_feat(filename):
    sample_rate = 16000
    samples, _ = librosa.load(filename, sr=sample_rate)
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


def load_tokens(args):
    ans = dict()
    with open(args.tokens_path, encoding="utf-8") as f:
        for line in f:
            sym, idx = line.strip().split()
            ans[int(idx)] = sym
    return ans


def main(args):
    filename = args.data_path
    features = compute_feat(filename)  # (T, C)
    features = np.expand_dims(features, axis=0)  # (N, T, C)
    features = features.transpose(0, 2, 1)  # (N, C, T)
    print("Features shape is ",features.shape)  # (N, C, T), (1, 80, 663)
    features_length = np.array([features.shape[2]], dtype=np.int64)
    print("Featurs length input: ",features_length)
    sess = ort.InferenceSession(args.model_path,providers=['CUDAExecutionProvider'])#, sess_options=sess_options, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']

    print("input is:")
    for n in sess.get_inputs():
        print(n.name, n.type, n.shape)

    print("output is:")
    for n in sess.get_outputs():
        print(n.name, n.type, n.shape)

    inputs = {
        sess.get_inputs()[0].name: features,
        sess.get_inputs()[1].name: features_length,
    }
    t1 = time.time()
    outputs = sess.run([sess.get_outputs()[0].name], input_feed=inputs)
    t2 = time.time()
    print("=========>time with profiling", t2-t1)
    # outputs[0] contains log_probs

    print("output shape is ",outputs[0].shape)  # (N, T, C), (1, 166, 1025)
    print("output dtype is ",outputs[0].dtype)  # float32
    print(np.exp(outputs[0]).sum(axis=-1).reshape(-1)[:10])  # validate it is log_probs
    indexes = outputs[0].argmax(axis=-1)
    print(indexes.shape)
    indexes = indexes.squeeze().tolist()
    unique_indexes = [k for k, _ in itertools.groupby(indexes)]
    print(indexes)
    print(unique_indexes)

    tokens = load_tokens(args)
    text = "".join([tokens[i] for i in unique_indexes if i != 1024])

    print("text is ",text)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)