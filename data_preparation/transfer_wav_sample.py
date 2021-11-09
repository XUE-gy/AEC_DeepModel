import librosa
import soundfile as sf


echo_signal = 12
farend_speech = 21
nearend_mic_signal = 19
nearend_speech = 22

import os, wave, audioop


def mul_stereo(sample, width, lfactor, rfactor):
    lsample = audioop.tomono(sample, width, 1, 0)
    rsample = audioop.tomono(sample, width, 0, 1)
    lsample = audioop.mul(lsample, width, lfactor)
    rsample = audioop.mul(rsample, width, rfactor)
    lsample = audioop.tostereo(lsample, width, 1, 0)
    rsample = audioop.tostereo(rsample, width, 0, 1)
    return audioop.add(lsample, rsample, width)

def downsampleWav(src, dst, inrate=44100, outrate=16000, inchannels=2, outchannels=2):

    if not os.path.exists(src):
        print('Source not found!')
        return False

    if not os.path.exists(os.path.dirname(dst)):
        os.makedirs(os.path.dirname(dst))

    try:
        s_read = wave.open(src, 'r')
        s_write = wave.open(dst, 'w')
    except:
        print('Failed to open files!')
        return False

    n_frames = s_read.getnframes()
    data = s_read.readframes(n_frames)

    try:
        converted = audioop.ratecv(data, 2, inchannels, inrate, outrate, None)
        if outchannels == 1:
            converted = audioop.tomono(converted[0], 2, 1, 0)
        if outchannels == 2:
            mul_stereo(converted[0], 2, 1, 1)
    except:
        print('Failed to downsample wav')
        return False

    try:
        s_write.setparams((outchannels, 2, outrate, 0, 'NONE', 'Uncompressed'))
        s_write.writeframes(converted)
    except:
        print('Failed to write wav')
        return False

    try:
        s_read.close()
        s_write.close()
    except:
        print('Failed to close wav files')
        return False

    return True

for i in range(0,491):

    # name = "echo_fileid_0"+str( "%04d"%(i) )+".wav"

    inputPath = "E:\\pycharmProject\\MyModel\AEC_DeepModel\\data_preparation\\Synthetic\\MultiTRAIN44100\\nearend_mic_signal\\"
    outputPath = "E:\\pycharmProject\\MyModel\AEC_DeepModel\\data_preparation\\Synthetic\\MultiTRAIN16000\\nearend_mic_signal\\"

    name = inputPath + "nearend_mic_signal_0_" + str(i) + ".wav"
    outputName = outputPath + "nearend_mic_signal_0_" + str(i) + ".wav"

    print(i)
    # name = ""
    src_sig,sr = sf.read(name)  #name是要 输入的wav 返回 src_sig:音频数据  sr:原采样频率
    # print(src_sig)
    dst_sig = librosa.resample(src_sig,sr,16000)  #resample 入参三个 音频数据 原采样频率 和目标采样频率
    sf.write(outputName,dst_sig,16000) #写出数据  参数三个 ：  目标地址  更改后的音频数据  目标采样数据

    # downsampleWav(name,outputName)





# inputPath = "E:\\TrainWav\\TRAIN2_2\\farend_speech\\"
# outputPath = "E:\\TrainWav\\TRAIN2_2\\farend_speech\\"
#
# name = inputPath + "playlist.wav"
# outputName = outputPath + "playlist44100.wav"
# # name = ""
# src_sig,sr = sf.read(name)  #name是要 输入的wav 返回 src_sig:音频数据  sr:原采样频率
# # print(src_sig)
# print(sr)
# dst_sig = librosa.resample(src_sig,sr,44100)  #resample 入参三个 音频数据 原采样频率 和目标采样频率
# sf.write(outputName,dst_sig,44100) #写出数据  参数三个 ：  目标地址  更改后的音频数据  目标采样数据
