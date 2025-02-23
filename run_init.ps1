# Stop execution immediately if any command fails
$ErrorActionPreference = "Stop"

# Function to write colored output with emojis
function Write-EmojiMessage {
    param (
        [string]$Message,
        [ConsoleColor]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

Write-EmojiMessage "🚀 Starting initialization..." Cyan

# 1. Add the git submodule
Write-EmojiMessage "📥 Adding git submodule..." Yellow
git submodule add https://github.com/DepthAnything/Depth-Anything-V2 src/third_party
Write-EmojiMessage "✅ Git submodule added successfully!" Green
Write-Host ""

# 2. Install pip requirements
Write-EmojiMessage "📦 Installing pip requirements..." Yellow
pip install -r requirements.txt
Write-EmojiMessage "✅ Pip requirements installed!" Green
Write-Host ""

# 3. Install PyTorch, torchvision, and torchaudio with CUDA 12.6 support
Write-EmojiMessage "🛠 Installing PyTorch, torchvision, and torchaudio..." Yellow
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
Write-EmojiMessage "✅ PyTorch packages installed!" Green
Write-Host ""

# 4. Install NVIDIA cuDNN for CUDA 12
Write-EmojiMessage "🔧 Installing NVIDIA cuDNN..." Yellow
py -m pip install nvidia-cudnn-cu12
Write-EmojiMessage "✅ NVIDIA cuDNN installed!" Green
Write-Host ""

Write-EmojiMessage "🎉 Initialization complete!" Cyan
