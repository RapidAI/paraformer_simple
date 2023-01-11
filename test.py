from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_16k_pipline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',device='cpu')

import time
start=time.time()
print('start',start)
for i in range(10):
  rec_result = inference_16k_pipline(audio_in='./asr_example_zh.wav')
  print(rec_result,time.time()-start)
print(time.time()-start)
