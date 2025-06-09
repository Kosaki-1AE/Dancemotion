from pydub import AudioSegment
from pydub.playback import play

# 音階とファイル名の対応表
notes = {
    "C": "C.wav",
    "C#": "C#.wav",
    "D": "D.wav",
    "D#": "D#.wav",
    "E": "E.wav",
    "F": "F.wav",
    "F#": "F#.wav",
    "G": "G.wav",
    "G#": "G#.wav",
    "A": "A.wav",
    "A#": "A#.wav",
    "B": "B.wav"
}

def play_note(note):
    filename = notes.get(note)
    if filename:
        sound = AudioSegment.from_wav(f"scales/{filename}")
        play(sound)
    else:
        print("未対応の音階です！")

# 実行例
if __name__ == "__main__":
    note_input = input("CDEFGABで入力してね：")
    play_note(note_input.strip())
