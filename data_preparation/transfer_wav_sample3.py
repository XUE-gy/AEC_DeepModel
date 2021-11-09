import librosa
import soundfile as sf


echo_signal = 12
farend_speech = 21
nearend_mic_signal = 19
nearend_speech = 22
for i in range(1000,1001):

    # name = "echo_fileid_0"+str( "%04d"%(i) )+".wav"

    inputPath = "E:\\TrainWav\\TRAIN2\\nearend_mic_signal\\"
    outputPath = "E:\\TrainWav\\TRAIN2\\nearend_mic_signal2\\"

    name = inputPath + "nearend_mic_fileid_" + str(i) + ".wav"
    outputName = outputPath + "nearend_mic_fileid_" + str(i) + ".wav"
    # name = ""
    src_sig,sr = sf.read(name)  #name是要 输入的wav 返回 src_sig:音频数据  sr:原采样频率
    # print(src_sig)
    print(i)
    dst_sig = librosa.resample(src_sig,sr,44100)  #resample 入参三个 音频数据 原采样频率 和目标采样频率
    sf.write(outputName,dst_sig,44100) #写出数据  参数三个 ：  目标地址  更改后的音频数据  目标采样数据





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
