from pydub import AudioSegment
from pydub.playback import play

# 音階とファイル名の対応表
notes = {
    "C": "C.wav",
    "a_F1": "a_F1.wav",
    "i_F1": "i_F1.wav",
    "u_F1": "u_F1.wav",
    "e_F1": "e_F1.wav",
    "o_F1": "o_F1.wav",
    "a_F2": "a_F2.wav",
    "i_F2": "i_F2.wav",
    "u_F2": "u_F2.wav",
    "e_F2": "e_F2.wav",
    "o_F2": "o_F2.wav",
    "a_F3": "a_F3.wav",
    "i_F3": "i_F3.wav",
    "u_F3": "u_F3.wav",
    "e_F3": "e_F3.wav",
    "o_F3": "o_F3.wav",
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
