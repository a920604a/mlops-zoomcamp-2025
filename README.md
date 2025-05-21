# mlops-zoomcamp-2025



## 下載並安裝 Python 的 Anaconda 發行版
```
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
bash Anaconda3-2022.05-Linux-x86_64.sh
```
```
curl -O https://repo.anaconda.com/archive/Anaconda3-2022.05-MacOSX-x86_64.sh
bash Anaconda3-2022.05-MacOSX-x86_64.sh
echo 'export PATH="/Users/chenyuan/anaconda3/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
conda create --name mlops python=3.11
```