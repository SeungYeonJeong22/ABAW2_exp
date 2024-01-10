# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Compute input examples for VGGish from audio waveform."""

import numpy as np
import resampy

from vggish import mel_features
from vggish import vggish_params

try:
  import soundfile as sf

  def wav_read(wav_file):
    wav_data, sr = sf.read(wav_file, dtype='int16')
    return wav_data, sr

except ImportError:

  def wav_read(wav_file):
    raise NotImplementedError('WAV file reading requires soundfile package.')


def waveform_to_examples(data, sample_rate, window_sec, hop_sec):
  """Converts audio waveform into an array of examples for VGGish.
  Args:
    data: np.array of either one dimension (mono) or two dimensions
      (multi-channel, with the outer dimension representing channels).
      Each sample is generally expected to lie in the range [-1.0, +1.0],
      although this is not required.
    sample_rate: Sample rate of data.
  Returns:
    3-D np.array of shape [num_examples, num_frames, num_bands] which represents
    a sequence of examples, each of which contains a patch of log mel
    spectrogram, covering num_frames frames of audio and num_bands mel frequency
    bands, where the frame length is vggish_params.STFT_HOP_LENGTH_SECONDS.
  """
  # Convert to mono.
  if len(data.shape) > 1:
    data = np.mean(data, axis=1)
  # Resample to the rate assumed by VGGish.
  if sample_rate != vggish_params.SAMPLE_RATE:
    data = resampy.resample(data, sample_rate, vggish_params.SAMPLE_RATE)
    
  print("after resample data : ", data.shape)
  print("window_sec, hop_sec : ", window_sec, hop_sec)

  # Compute log mel spectrogram features.
  log_mel = mel_features.log_mel_spectrogram(
      data,
      audio_sample_rate=vggish_params.SAMPLE_RATE,
      log_offset=vggish_params.LOG_OFFSET,
      window_length_secs=vggish_params.STFT_WINDOW_LENGTH_SECONDS,
      hop_length_secs=vggish_params.STFT_HOP_LENGTH_SECONDS,
      num_mel_bins=vggish_params.NUM_MEL_BINS,
      lower_edge_hertz=vggish_params.MEL_MIN_HZ,
      upper_edge_hertz=vggish_params.MEL_MAX_HZ)
  
  # x: frame size, y: Mel 주파수 영역의 여러 밴드
  # 
  
  print("log_mel : ", log_mel.shape)

  # Frame features into examples.
  features_sample_rate = 1.0 / vggish_params.STFT_HOP_LENGTH_SECONDS
  example_window_length = int(round(
      window_sec * features_sample_rate))

  # example_hop_length = int(round(
  #     hop_sec * features_sample_rate))

  example_hop_length = hop_sec * features_sample_rate
  
  num_samples = log_mel.shape[0]
  num_frames = 1 + int(np.floor((num_samples - example_window_length) / example_hop_length))
  
  while num_frames > 5500:
    hop_sec += 0.01
    example_hop_length = hop_sec * features_sample_rate
    num_frames = 1 + int(np.floor((num_samples - example_window_length) / example_hop_length))
  
  log_mel_examples = mel_features.my_frame(
      log_mel,
      num_frames,
      window_length=example_window_length,
      hop_length=example_hop_length)
  return log_mel_examples


def wavfile_to_examples(wav_file, window_sec, hop_sec):
  """Convenience wrapper around waveform_to_examples() for a common WAV format.
  Args:
    wav_file: String path to a file, or a file-like object. The file
    is assumed to contain WAV audio data with signed 16-bit PCM samples.
  Returns:
    See waveform_to_examples.
  """
  wav_data, sr = wav_read(wav_file)
  assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
  
  # 32768 == 16비트 wav파일에 대한 정규화 값 2^16 -1
  samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]
  # print("wav_data : ", wav_data)
  # print("wav_data type : ", type(wav_data))
  # print("wav_data length : ", len(wav_data))
  
  # # -1보다 작거나 1보다 큰 값을 확인
  # out_of_range_samples = samples[(samples < -1.0) | (samples > 1.0)]

  # # 결과 출력
  # if len(out_of_range_samples) == 0:
  #     print("모든 샘플 값이 올바른 범위에 있습니다.")
  # else:
  #     print(f"경고: {len(out_of_range_samples)} 개의 샘플이 올바른 범위를 벗어났습니다.")
  #     print("최소값:", np.min(samples))
  #     print("최대값:", np.max(samples))
  #     print("올바른 범위를 벗어난 값들:", out_of_range_samples)

  samples = np.pad(samples, (0, sr), 'edge')
  return waveform_to_examples(samples, sr, window_sec, hop_sec)