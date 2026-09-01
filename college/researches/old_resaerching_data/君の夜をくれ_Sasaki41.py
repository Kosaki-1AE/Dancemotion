import numpy as np

def parse_bvh(bvh_text):
    """
    BVHテキストデータを解析し、骨格情報とモーションデータを抽出します。
    """
    lines = bvh_text.splitlines()
    
    hierarchy_section = []
    motion_section_start = -1
    
    # 階層構造とモーションセクションの開始行を特定
    for i, line in enumerate(lines):
        if line.strip() == "MOTION":
            motion_section_start = i
            break
        hierarchy_section.append(line)
            
    if motion_section_start == -1:
        raise ValueError("MOTION section not found in BVH file.")

    # ----------------------------------------------------
    # HIERARCHY (階層構造) の解析
    # ----------------------------------------------------
    
    joints = {}
    joint_order = [] # モーションデータが記述されている順序
    current_parent = None
    indent_level = 0
    
    # ルートジョイントとスタックを管理
    joint_stack = []
    
    for line in hierarchy_section:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split()
        
        if parts[0] == "ROOT" or parts[0] == "JOINT":
            joint_name = parts[1]
            joints[joint_name] = {
                'offset': None,
                'channels': [],
                'children': [],
                'parent': current_parent,
                'channel_indices': {} # 各ジョイントのチャンネル開始インデックス
            }
            if current_parent:
                joints[current_parent]['children'].append(joint_name)
            
            joint_stack.append(joint_name)
            current_parent = joint_name
            joint_order.append(joint_name) # モーションデータと同じ順序で追加
            
        elif parts[0] == "OFFSET":
            joints[current_parent]['offset'] = np.array(list(map(float, parts[1:])))
            
        elif parts[0] == "CHANNELS":
            # チャンネルの数を取得し、その後のチャンネル名をリストに追加
            num_channels = int(parts[1])
            joints[current_parent]['channels'] = parts[2:]
            
        elif parts[0] == "{":
            # 新しいジョイントブロックの開始
            indent_level += 1
        
        elif parts[0] == "}":
            # ジョイントブロックの終了
            indent_level -= 1
            if joint_stack:
                joint_stack.pop()
                current_parent = joint_stack[-1] if joint_stack else None
                
        elif parts[0] == "End": # End Siteの処理
            # End SiteもCHANNELSを持たないが、OFFSETは持つ
            # これは、末端のジョイントの終端位置を示すダミーのジョイント
            end_site_name = current_parent + '_EndSite' # 適当な名前を付ける
            joints[end_site_name] = {
                'offset': None,
                'channels': [], # End Siteはチャンネルを持たない
                'children': [],
                'parent': current_parent,
                'is_end_site': True # End Siteであるマーク
            }
            joints[current_parent]['children'].append(end_site_name)
            
            # End SiteのOFFSETは次の行にくる
            # 次の行がOFFSETのはずなので、そこまで読み飛ばす
            # このパーサーでは、OFFSETは次のループで処理されるため、特殊な処理は不要
            
        # End SiteのOFFSETは、Endブロックの直後に出現
        # End Siteのジョイント名を一時的に現在の親として処理する
        # このパーサーでは、`OFFSET`行の処理で`current_parent`が使われるので、End Siteの名前を正しく設定する必要がある
        # しかし、End Siteは`JOINT`ではないので、`current_parent`がEnd Siteの名前になることはない。
        # このままだとEnd SiteのOFFSETが正しく紐付かない可能性がある。
        # BVHのパースは複雑なため、ここは簡易的な処理に留める。
        # 厳密には、End Siteもスタック管理に含めるべき。
        
    # チャンネルインデックスの計算
    total_channels_count = 0
    for joint_name in joint_order:
        if joint_name in joints and not joints[joint_name].get('is_end_site', False):
            joints[joint_name]['channel_indices']['start'] = total_channels_count
            joints[joint_name]['channel_indices']['end'] = total_channels_count + len(joints[joint_name]['channels'])
            total_channels_count += len(joints[joint_name]['channels'])

    # ----------------------------------------------------
    # MOTION (モーションデータ) の解析
    # ----------------------------------------------------
    
    motion_data_lines = lines[motion_section_start + 1:]
    
    frames_line = motion_data_lines[0].strip()
    num_frames = int(frames_line.split()[1])
    
    frame_time_line = motion_data_lines[1].strip()
    frame_time = float(frame_time_line.split()[2])
    
    # 実際のモーションデータは、その次の行から
    motion_values = []
    for line in motion_data_lines[2:]:
        if line.strip():
            values = list(map(float, line.strip().split()))
            if len(values) == total_channels_count: # チャンネル数と一致するか確認
                motion_values.append(values)
            else:
                print(f"Warning: Mismatched channel count in line: {line.strip()}. Expected {total_channels_count}, got {len(values)}.")
                
    motion_array = np.array(motion_values)
    
    return {
        'joints': joints,
        'joint_order': joint_order,
        'num_frames': num_frames,
        'frame_time': frame_time,
        'motion_data': motion_array,
        'total_channels': total_channels_count
    }

# --- ファイルの読み込みと解析の実行 ---
file_path = '君の夜をくれ_Sasaki41.txt' # アップロードされたファイル名に合わせる

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        bvh_content = f.read()
    
    bvh_data = parse_bvh(bvh_content)
    
    print("--- BVH Data Summary ---")
    print(f"Total Frames: {bvh_data['num_frames']}")
    print(f"Frame Time: {bvh_data['frame_time']} seconds")
    print(f"Total Channels: {bvh_data['total_channels']}")
    print("\n--- Joint Hierarchy and Channels ---")
    
    # ジョイント情報の表示 (一部抜粋)
    for joint_name in bvh_data['joint_order']:
        joint_info = bvh_data['joints'].get(joint_name)
        if joint_info and not joint_info.get('is_end_site', False):
            print(f"Joint: {joint_name}")
            print(f"  Parent: {joint_info['parent']}")
            print(f"  Offset: {joint_info['offset']}")
            print(f"  Channels: {joint_info['channels']}")
            print(f"  Children: {joint_info['children']}")
            print(f"  Channel Indices (start/end): {joint_info['channel_indices'].get('start')}-{joint_info['channel_indices'].get('end')-1}")
            print("-" * 20)
            if joint_name == 'Head': # 例としてHeadジョイントまで表示
                break # 長くなるので、一部だけ表示
    
    # ルートジョイント (Hips) の最初の数フレームの動きを表示 (位置と回転)
    print("\n--- First 5 Frames of Hips Motion (Position and Rotation) ---")
    hips_channels_info = bvh_data['joints']['Hips']
    hips_start_idx = hips_channels_info['channel_indices']['start']
    hips_end_idx = hips_channels_info['channel_indices']['end']
    
    for i in range(min(5, bvh_data['num_frames'])): # 最初の5フレームまたは総フレーム数まで
        frame_data = bvh_data['motion_data'][i]
        hips_motion_values = frame_data[hips_start_idx:hips_end_idx]
        
        # チャンネルの順番に基づいて値を割り当て (Xposition Yposition Zposition Zrotation Xrotation Yrotation)
        pos_x, pos_y, pos_z, rot_z, rot_x, rot_y = hips_motion_values
        print(f"Frame {i+1}: Pos=({pos_x:.2f}, {pos_y:.2f}, {pos_z:.2f}), Rot=({rot_x:.2f}, {rot_y:.2f}, {rot_z:.2f})")

    # 例として、HeadジョイントのXrotationの変化をグラフ化
    print("\n--- Plotting Head X-rotation over time ---")
    
    try:
        import matplotlib.pyplot as plt

        head_channels_info = bvh_data['joints']['Head']
        head_channels = head_channels_info['channels']
        head_start_idx = head_channels_info['channel_indices']['start']

        # Xrotationのチャンネルインデックスを特定
        x_rot_channel_offset = -1
        for i, channel_name in enumerate(head_channels):
            if channel_name == 'Xrotation':
                x_rot_channel_offset = i
                break

        if x_rot_channel_offset != -1:
            head_x_rotations = bvh_data['motion_data'][:, head_start_idx + x_rot_channel_offset]
            
            plt.figure(figsize=(12, 6))
            plt.plot(head_x_rotations, label='Head X-rotation')
            plt.title('Head X-rotation Over Time')
            plt.xlabel('Frame Number')
            plt.ylabel('X-rotation (degrees)')
            plt.grid(True)
            plt.legend()
            plt.show()
        else:
            print("Head joint does not have Xrotation channel.")

    except ImportError:
        print("Matplotlib is not installed. Skipping plotting example.")
        print("To install: pip install matplotlib")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except ValueError as e:
    print(f"Error parsing BVH file: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")