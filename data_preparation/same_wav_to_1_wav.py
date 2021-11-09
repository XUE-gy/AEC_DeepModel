# python 音频文件处理
# 多个 wav 格式语音文件合成
from pydub import AudioSegment
import os
import shutil
import time

AudioSegment.converter = "D:\\dependence\\ffmpeg-20191013-4f4334b-win64-static\\bin\\ffmpeg.exe"

# 每个文件夹音频的前缀长
echo_signal = 12
farend_speech = 21
nearend_mic_signal = 19
nearend_speech = 22
new_dir = 'E:\\TrainWav\\TRAIN2\\echo_signal2\\'  # wav语音文件所在文件夹
output_dir = 'E:\\TrainWav\\TRAIN2_2\\echo_signal2\\'  # wav输出语音文件所在文件夹
list_voice_dir = os.listdir(new_dir)
list_voice_dir.sort(key=lambda x: int(x[echo_signal:][:-4]))  # 先取12位之后，再倒着数第四位'.'为分界线，按照‘.'左边的数字从小到大


# wav 格式语音文件合成
def voice_unit():
    n = 0
    # list_voice_dir_length = len(list_voice_dir)
    playlist = AudioSegment.empty()
    second_5_silence = AudioSegment.silent(duration=1000)  # 产生一个持续时间为1s的无声AudioSegment对象
    for i in list_voice_dir:
        print(n)
        # sound = AudioSegment.from_wav(list_voice_dir[n])
        sound = AudioSegment.from_file(new_dir + list_voice_dir[n], format="wav")  # wav
        # sound = AudioSegment.from_file(new_dir+list_voice_dir[n],sample_width=2,frame_rate=16000,channels=1)  #raw pcm
        playlist = playlist + sound + second_5_silence
        n += 1
    playlist.export(output_dir + 'playlist44100.wav', format="wav")  # wav
    # playlist.export(new_dir+'playlist.pcm')   #pcm
    print("语音合成完成，合成文件放在：", new_dir, "目录下")


# 对比文件顺序
def testlist():
    for i in list_voice_dir:
        voicename = i
        print("对比文件顺序是否改变：", voicename)


def main():
    # for i in list_voice_dir:
    #     print(i)
    try:
        os.remove(new_dir + 'playlist.pcm')
    except:
        print("")
    testlist()
    voice_unit()


if __name__ == "__main__":
    main()
