import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

# =====================================================
# Model Definition
# =====================================================

class Encoder(nn.Module):
    def __init__(self, latent_dim=3, num_classes=10):
        super().__init__()

        self.label_embedding = nn.Linear(num_classes, 16)
        self.fc_hidden = nn.Linear(28 * 28 + 16, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, image, label):
        flattened_image = image.view(image.size(0), -1)

        label_one_hot = F.one_hot(label, num_classes=10).float()
        label_embedding = F.relu(self.label_embedding(label_one_hot))

        x = torch.cat([flattened_image, label_embedding], dim=1)
        hidden = F.relu(self.fc_hidden(x))

        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=3, num_classes=10):
        super().__init__()

        self.label_embedding = nn.Linear(num_classes, 16)
        self.fc_hidden = nn.Linear(latent_dim + 16, 128)
        self.fc_out = nn.Linear(128, 28 * 28)

    def forward(self, latent_vector, label):
        label_one_hot = F.one_hot(label, num_classes=10).float()
        label_embedding = F.relu(self.label_embedding(label_one_hot))

        x = torch.cat([latent_vector, label_embedding], dim=1)
        hidden = F.relu(self.fc_hidden(x))

        output = torch.sigmoid(self.fc_out(hidden))

        return output.view(-1, 1, 28, 28)


class CVAE(nn.Module):
    def __init__(self, latent_dim=3, num_classes=10):
        super().__init__()

        self.encoder = Encoder(latent_dim, num_classes)
        self.decoder = Decoder(latent_dim, num_classes)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, image, label):
        mu, logvar = self.encoder(image, label)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z, label)

        return reconstructed, mu, logvar


# =====================================================
# Device
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================
# Load Model
# =====================================================

@st.cache_resource
def load_model():
    model = CVAE(latent_dim=3, num_classes=10).to(device)

    state_dict = torch.load(
        "cvae.pth",
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


model = load_model()


# =====================================================
# Streamlit UI
# =====================================================

st.title("Conditional Variational Autoencoder")
st.write("学習済みCVAEを用いてMNIST数字画像を生成します。")

digit = st.selectbox(
    "生成したい数字",
    list(range(10))
)

n_samples = st.slider(
    "生成枚数",
    min_value=1,
    max_value=25,
    value=9
)

if st.button("画像生成"):

    latent_dim = 3

    # 潜在変数を生成
    z = torch.randn(n_samples, latent_dim).to(device)

    # ラベルを作成
    labels = torch.full(
        (n_samples,),
        digit,
        dtype=torch.long,
        device=device
    )

    # 推論
    with torch.no_grad():
        generated = model.decoder(z, labels)

    # 表示レイアウト
    cols = math.ceil(math.sqrt(n_samples))
    rows = math.ceil(n_samples / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    # axesを2次元配列に統一
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    # 描画
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]

        if i < n_samples:
            ax.imshow(
                generated[i].squeeze().cpu().numpy(),
                cmap="gray"
            )

        ax.axis("off")

    plt.tight_layout()

    st.pyplot(fig)
