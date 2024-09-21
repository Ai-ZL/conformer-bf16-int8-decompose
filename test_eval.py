import numpy as np
import kaldi_native_fbank as knf
import itertools
import onnxruntime as ort
from datasets import load_dataset
from evaluate import load
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-model', '--model_path', help='model stored path')
parser.add_argument('-t', '--tokens_path', help='tokens list stored path')

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

def load_tokens(args):
    ans = dict()
    with open(args.tokens_path, encoding="utf-8") as f:
        for line in f:
            sym, idx = line.strip().split()
            ans[int(idx)] = sym
    return ans

def main(args):
    librispeech_eval = load_dataset("librispeech_asr", "clean", split="test")  # change to "other" for other test dataset
    wer = load("wer")

    sess = ort.InferenceSession(args.model_path,providers=['CUDAExecutionProvider'])

    WER_s=0
    larger_s=0
    larger_s_list=[]
    larger_s_wer_list=[]
    for i in range(len(librispeech_eval)):
        features = compute_feat(librispeech_eval[i]['audio']['array'],
                            librispeech_eval[i]['audio']['sampling_rate'])  # (T, C)
        features = np.expand_dims(features, axis=0)  # (N, T, C)
        features = features.transpose(0, 2, 1)  # (N, C, T)
        features_length = np.array([features.shape[2]], dtype=np.int64)
        inputs = {
            sess.get_inputs()[0].name: features,
            sess.get_inputs()[1].name: features_length,
        }

        outputs = sess.run([sess.get_outputs()[0].name], input_feed=inputs)

        indexes = outputs[0].argmax(axis=-1)
        indexes = indexes.squeeze().tolist()
        unique_indexes = [k for k, _ in itertools.groupby(indexes)]

        tokens = load_tokens(args)
        text = "".join([tokens[i] for i in unique_indexes if i != 1024])
        text = text.replace("▁", " ");
        text = text[1:]
        text = text.upper()
        print(text)
        print(librispeech_eval[i]['text'])
        predictions = [text]
        references = [librispeech_eval[i]['text']]
        wer_score = wer.compute(predictions=predictions, references=references)
        print(wer_score)
        WER_s+=wer_score

        if wer_score>0.5:
            larger_s+=1
            larger_s_list.append(i)
            larger_s_wer_list.append(wer_score)
        print("wer larger than 0.5 is ",larger_s)

    print("dataset size: ",len(librispeech_eval))
    print("average WER score is: ",WER_s/len(librispeech_eval))
    print("wer larger than 0.5: ",larger_s)
    print("larger than 0.5 list is ",larger_s_list)
    print("larger than 0.5 list wer is ",larger_s_wer_list)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)