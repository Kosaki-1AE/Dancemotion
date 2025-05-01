// mocopi + VIVE を Unity 上でリアルタイム統合するシンプル構成

using UnityEngine;
using Valve.VR; // SteamVR 用

public class RealTimeMocopiVive : MonoBehaviour
{
    // --- VIVE 側（SteamVR） ---
    public SteamVR_Input_Sources headSource = SteamVR_Input_Sources.Head;
    public Transform headTransform;

    public SteamVR_Input_Sources leftHandSource = SteamVR_Input_Sources.LeftHand;
    public Transform leftHandTransform;

    public SteamVR_Input_Sources rightHandSource = SteamVR_Input_Sources.RightHand;
    public Transform rightHandTransform;

    // --- mocopi 側 ---
    [System.Serializable]
    public class MocopiBone
    {
        public string boneName;
        public Transform targetTransform;
    }
    public MocopiBone[] mocopiBones;

    // --- OSC/UDPなどで mocopi データを受信する想定（簡易化のため自動更新は未実装） ---
    private Dictionary<string, Quaternion> mocopiRotations = new Dictionary<string, Quaternion>();

    void Update()
    {
        // VIVE のトラッキング座標を反映
        if (headTransform != null)
            headTransform.localPosition = SteamVR_Input_Pose.GetLocalPosition(headSource);

        if (leftHandTransform != null)
            leftHandTransform.localPosition = SteamVR_Input_Pose.GetLocalPosition(leftHandSource);

        if (rightHandTransform != null)
            rightHandTransform.localPosition = SteamVR_Input_Pose.GetLocalPosition(rightHandSource);

        // mocopi ボーンの回転を反映
        foreach (var bone in mocopiBones)
        {
            if (mocopiRotations.TryGetValue(bone.boneName, out Quaternion rot))
            {
                bone.targetTransform.localRotation = rot;
            }
        }
    }

    // mocopi からのデータ受信用（外部から呼び出す）
    public void SetMocopiRotation(string boneName, Quaternion rotation)
    {
        mocopiRotations[boneName] = rotation;
    }
} 

// 🔧使い方メモ：
// - Unity 上に Humanoid アバターを配置（Transform を mocopiBones[] に割り当て）
// - VIVE 側の headTransform などには、カメラやコントローラのオブジェクトを指定
// - mocopi から UDP/OSC 経由で SetMocopiRotation() を呼び出す処理を別途実装する
