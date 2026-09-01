import torch
import torch.nn as nn
import torch.optim as optim
from improvformer import ImprovFormer, visualize_attention
from torch.utils.data import DataLoader, TensorDataset

# 仮のデータセット
batch_size = 8
seq_len = 10
input_dim = 51
output_dim = 10
num_classes = 4
x_data = torch.randn(100, seq_len, input_dim)
y_data = torch.randn(100, seq_len, output_dim)
labels = torch.randint(0, num_classes, (100,))

dataset = TensorDataset(x_data, y_data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# モデル準備
model = ImprovFormer(input_dim=input_dim, output_dim=output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# GPU対応（任意）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 学習ループ
for epoch in range(5):
    total_loss = 0
    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# モデル保存
torch.save(model.state_dict(), "improvformer_model.pth")

# 💡 Attention可視化（最後のバッチを例に）
print("\n🧠 Self-Attention 可視化中...")
model.eval()
with torch.no_grad():
    x_sample = x_data[:1].to(device)
    output, attn = model(x_sample, return_attention=True)
    visualize_attention()