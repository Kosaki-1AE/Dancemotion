import os

# txtファイルがあるフォルダ
folder_path = "E:/MyFeedbacks/Student_Age/College_Age/2ndGrade"
        
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        os.remove(file_path)
        print(f"{filename} を削除しました！")
