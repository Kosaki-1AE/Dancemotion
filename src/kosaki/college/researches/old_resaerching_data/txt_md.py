import os

# txtファイルがあるフォルダ
folder_path = "E:/MyFeedbacks/Student_Age/College_Age/4thGrade"

for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        # 拡張子なしの名前
        base_name = os.path.splitext(filename)[0]
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, base_name + ".md")

        os.rename(old_path, new_path)
        print(f"{filename} → {base_name}.md にリネームしました！")
        
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        os.remove(file_path)
        print(f"{filename} を削除しました！")
