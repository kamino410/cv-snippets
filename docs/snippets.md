1. [NeoVim & Terminal Setup](#sec2)
    1. [Ubuntu 16.04](#sec2_1)
    1. [Ubuntu 18.04](#sec2_2)
    1. [Windows 10](#sec2_3)
1. [Git Setup](#sec3)
1. [Docker](#sec_docker)
    1. [Enable ctrl+p](#sec_docker_ctrlp)
    1. [GUI via XWindow](#sec_docker_gui)
    1. [Jupyter Notebook on Container](#sec_docker_notebook)
1. [Install OpenCV](#sec_opencv)
1. [Install Latest CMake into Ubuntu](#sec4)
1. [CMake Templates](#sec5)
    1. [Base](#sec5_1)
    1. [OpenCV](#sec5_2)
    1. [Eigen](#sec5_3)
    1. [Ceres](#sec5_4)
    1. [CUDA](#sec5_5)
    1. [OpenMP](#sec5_6)
    1. [Threads](#sec5_7)
1. [C++](#sec_cpp)
    1. [Stopwatch](#sec_cpp_1)
    1. [CSV Reader](#sec_cpp_2)
    1. [OpenCV Standard Operations](#sec_cpp_opencv)
    1. [OpenCV File Storage](#sec_cpp_filestorage)
    1. [OpenCV Fullscreen](#sec_cpp_fullscreen)
1. [Python](#sec_py)
    1. [argparse Template](#sec_py_argparse)
    1. [CSV Reader](#sec_py_1)
    1. [Simple XML Writer/Reader](#sec_py_2)
    1. [OpenCV File Storage](#sec_py_filestorage)
    1. [OpenCV Fullscreen](#sec_py_fullscreen)
    1. [Plotly Simple Template](#sec_py_plotly)
    1. [Plotly 1D data](#sec_py_3)
    1. [Plotly Figure](#sec_py_plotly_figure)
1. [SIMD](#sec_simd)
    1. [Check supported architecture](#sec_simd_check)
    1. [SSE](#sec_simd_sse)
    1. [AVX](#sec_simd_avx)
1. [CUDA](#sec_cuda)
    1. [Define Threads](#sec_cuda_dim)
    1. [Kernel Stopwatch](#sec_cuda_stopwatch)
1. [Image Processing](#sec_imgproc)
    1. [Generate checkerboard](#sec_imgproc_chess)
1. [Network](#sec_network)
    1. [Check used IPs](#sec_check_used_ips)
    2. [Temporal HTTP Server](#sec_http_server)

<h2 id="sec2">NeoVim & Terminal Setup</h2>

<h3 id="sec2_1">Ubuntu 16.04</h3>

```sh
cat <<EOF2 >> install.sh
apt update
apt install -y software-properties-common
add-apt-repository -y ppa:neovim-ppa/unstable
apt update
apt install -y neovim git clang curl wget build-essential libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev zlib1g-dev clang-format cmake
cd ~
wget https://www.python.org/ftp/python/3.7.4/Python-3.7.4.tgz
tar -xzf Python-3.7.4.tgz
cd ./Python-3.7.4
./configure --with-ensurepip --enable-optimizations
make altinstall
ln -sf /usr/local/bin/python3.7 /usr/bin/python3
ln -s /usr/share/pyshared/lsb_release.py /usr/local/lib/python3.7/site-packages/lsb_release.py
python3 -m pip install --upgrade pip
python3 -m pip install neovim neovim-remote autopep8
mkdir -p ~/.config
mkdir -p ~/.config/dein
mkdir -p ~/.config/nvim
mkdir -p ~/.config/nvim/ftplugin
mkdir -p ~/github
cd ~/github
git clone https://github.com/kamino410/dotfiles.git
cd ./dotfiles
git pull
cp ./.config/nvim/init.vim ~/.config/nvim
cp ./.config/nvim/ftplugin/cpp.vim ~/.config/nvim/ftplugin
cp ./.config/dein/plugins.toml ~/.config/dein
cp ./.config/dein/plugins_lazy.toml ~/.config/dein
echo set sh=bash >> ~/.config/nvim/init.vim
cd ~
wget https://github.com/peco/peco/releases/download/v0.5.3/peco_linux_386.tar.gz
tar -xzf peco_linux_386.tar.gz
cp peco_linux_386/peco /usr/local/bin
ln -s /usr/local/bin/peco /usr/bin/peco
cat <<EOF >> ~/.bashrc
export LC_ALL=C.UTF-8
alias nvrr='nvr --remote-tab'
peco-select-history() {
    declare l=$(HISTTIMEFORMAT= history | sort -k1,1nr | perl -ne 'BEGIN { my @lines = (); } s/^\s*\d+\s*//; $in=$_; if (!(grep {$in eq $_} @lines)) { push(@lines, $in); print $in; }' | peco --query "$READLINE_LINE")
    READLINE_LINE="$l"
    READLINE_POINT=${#l}
}
bind -x '"\C-r": peco-select-history'
EOF
EOF2
sh install.sh
```

<h3 id="sec2_2">Ubuntu 18.04</h3>

```sh
cat <<EOF2 >> install.sh
apt update
apt install -y software-properties-common
add-apt-repository -y ppa:neovim-ppa/unstable
apt update
apt install -y python3 neovim python3-pip git clang clang-tools curl clang-format peco
python3 -m pip install pynvim neovim neovim-remote autopep8
mkdir -p ~/.config
mkdir -p ~/.config/dein
mkdir -p ~/.config/nvim
mkdir -p ~/.config/nvim/ftplugin
mkdir -p ~/github
cd ~/github
git clone https://github.com/kamino410/dotfiles.git
cd ./dotfiles
git pull
cp ./.config/nvim/init.vim ~/.config/nvim
cp ./.config/nvim/ftplugin/cpp.vim ~/.config/nvim/ftplugin
cp ./.config/dein/plugins.toml ~/.config/dein
cp ./.config/dein/plugins_lazy.toml ~/.config/dein
echo set sh=bash >> ~/.config/nvim/init.vim
cd ~
cat <<EOF >> ~/.bashrc
export LC_ALL=C.UTF-8
alias nvrr='nvr --remote-tab'
peco-select-history() {
    declare l=$(HISTTIMEFORMAT= history | sort -k1,1nr | perl -ne 'BEGIN { my @lines = (); } s/^\s*\d+\s*//; $in=$_; if (!(grep {$in eq $_} @lines)) { push(@lines, $in); print $in; }' | peco --query "$READLINE_LINE")
    READLINE_LINE="$l"
    READLINE_POINT=${#l}
}
bind -x '"\C-r": peco-select-history'
EOF
EOF2
sh install.sh
```

<h3 id="sec2_3">Windows 10</h3>

```ps
md ~\AppData\Local\nvim\autoload
$uri = 'https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim'
(New-Object Net.WebClient).DownloadFile(
  $uri,
  $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath(
    "~\AppData\Local\nvim\autoload\plug.vim"
  )
)
```

`~/AppData/Local/nvim/init.vim`

なぜかdeinだとうまくいかないのでvimplugを使っている。

```vim
let g:python3_host_prog = "C:/Program Files/Python37/python"

call plug#begin('~/.vim/plugged')
Plug 'Shougo/deoplete.nvim', { 'do': 'UpdateRemotePlugins' }
Plug 'Shougo/context_filetype.vim'
Plug 'scrooloose/nerdtree', { 'on': ['NERDTreeToggle', 'NERDTreeTabsToggle'] }
Plug 'jistr/vim-nerdtree-tabs', { 'on': 'NERDTreeTabsToggle' }
Plug 'tomasr/molokai'
Plug 'vim-airline/vim-airline'
Plug 'myusuf3/numbers.vim'
Plug 'autozimu/LanguageClient-neovim', { 'branch': 'next', 'do': 'powershell -executionpolicy bypass -File install.ps1' }
call plug#end()

let g:deoplete#enable_at_startup = 1

let NERDTreeShowHidden=1
nnoremap <silent><C-K><C-S> :NERDTreeTabsToggle<CR>

set hidden
set signcolumn=yes
let g:LanguageClient_autoStart=1
let g:LanguageClient_serverCommands = {
        \ 'c': ['clangd', '-compile-commands-dir=' . getcwd() . '/build'],
        \ 'cpp': ['clangd', '-compile-commands-dir=' . getcwd() . '/build'],
        \ 'python': ['pyls'],
        \ 'rust': ['rustup', 'run', 'nightly', 'rls'],
    \ }

colorscheme molokai

set number

set tabstop=4
set softtabstop=4
set shiftwidth=4
set expandtab
set autoindent
set smartindent
set list
set listchars=tab:ﾂｻ-,trail:-

set clipboard+=unnamedplus

set visualbell
set t_vb=

noremap ; :
noremap : .
noremap . ;
nnoremap <C-h> gT
nnoremap <C-l> gt
noremap <Space>h ^
noremap <Space>l $
noremap <Space>t :tabnew<CR>:te<CR>

tnoremap <C-[> <C-\><C-n>

filetype plugin indent on
syntax on

set shell=powershell
```

<h2 id="sec3">Git Setup</h2>

```sh
git config --global alias.st status
git config --global alias.glog "log --all --graph --date=short --decorate=short --pretty=format:'%Cgreen%h %Creset%cd %Cblue%cn %Creset%s'"
git config --global user.email kamino.dev@gmail.com
git config --global user.name kamino410
git config --global core.editor nvim
```

<h2 id="sec_docker">Docker Tips</h2>

<h3 id="sec_docker_ctrlp">Enable ctrl+p</h3>

`~/.docker/config.json`

```json
{
  "detachKeys" : "ctrl-\\"
}
```

<h3 id="sec_docker_gui">GUI by XWindow</h3>

* `ssh -Y xxxx@xxx.xxx.xxx.xxx`
* `sudo docker run -it --gpus all -e DISPLAY=$DISPLAY --net host -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:/root/.Xauthority <image>`
* Test : `apt install x11-app`
* To display on host Ubuntu : `xhost +local:docker`

<h3 id="sec_docker_notebook">Jupyter Notebook on Container</h3>

```sh
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
```

<h2 id="sec_opencv">Install Opencv</h2>

必要に応じてコンパイルオプション・バージョンを変えること。
    
```sh
#!/bin/bash -e
myRepo=$(pwd)
CMAKE_CONFIG_GENERATOR="Visual Studio 15 2017 Win64"
if [  ! -d "$myRepo/opencv"  ]; then
    echo "clonning opencv"
    git clone https://github.com/opencv/opencv.git
    mkdir -p Build
    mkdir -p Build/opencv
    mkdir -p Install
    mkdir -p Install/opencv
fi
pushd opencv
git checkout master
git pull --rebase
git checkout refs/tags/4.0.1
popd

if [  ! -d "$myRepo/opencv_contrib"  ]; then
    echo "clonning opencv_contrib"
    git clone https://github.com/opencv/opencv_contrib.git
    mkdir -p Build
    mkdir -p Build/opencv_contrib
fi
pushd opencv_contrib
git checkout master
git pull --rebase
git checkout refs/tags/4.0.1
popd

RepoSource=opencv
pushd Build/$RepoSource
CMAKE_OPTIONS='-DBUILD_PERF_TESTS:BOOL=OFF -DBUILD_TESTS:BOOL=OFF -DBUILD_DOCS:BOOL=OFF  -DWITH_CUDA:BOOL=OFF -DBUILD_EXAMPLES:BOOL=OFF -DINSTALL_CREATE_DISTRIB=ON'
cmake -G"$CMAKE_CONFIG_GENERATOR" $CMAKE_OPTIONS -DOPENCV_EXTRA_MODULES_PATH="$myRepo"/opencv_contrib/modules -DCMAKE_INSTALL_PREFIX="$myRepo"/install/"$RepoSource" "$myRepo/$RepoSource"
echo "************************* $Source_DIR -->debug"
cmake --build .  --config debug
echo "************************* $Source_DIR -->release"
cmake --build .  --config release
cmake --build .  --target install --config release
cmake --build .  --target install --config debug
popd
```

<h2 id="sec4">Install Latest CMake into Ubuntu</h2>

適宜バージョンを指定すること。

```sh
CMAKE_VER=3.18.4
apt update
apt install -y libncurses5-dev libcurl4-openssl-dev zlib1g-dev wget
wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VER}/cmake-${CMAKE_VER}.tar.gz
tar -xzf cmake-${CMAKE_VER}.tar.gz
cd cmake-${CMAKE_VER}
./bootstrap --system-curl
make install -j7
ln -s /usr/local/bin/cmake /usr/bin/cmake
ln -s /usr/local/bin/ccmake /usr/bin/ccmake
```

<h2 id="sec5">CMake Templates</h2>

<h3 id="sec5_1">Base</h3>

```cmake
cmake_minimum_required(VERSION 3.1)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(main CXX)

add_executable(main main.cpp)
```

<h3 id="sec5_2">OpenCV</h3>

```cmake
find_package(OpenCV REQUIRED)
target_include_directories(main PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(main ${OpenCV_LIBS})
```

<h3 id="sec5_3">Eigen</h3>

```cmake
find_package(Eigen3 REQUIRED)
target_include_directories(main PUBLIC ${EIGEN3_INCLUDE_DIRS})
```

<h3 id="sec5_4">Ceres</h3>

```cmake
find_package(Ceres REQUIRED)
target_include_directories(main PUBLIC ${CERES_INCLUDE_DIRS})
target_link_libraries(main ${CERES_LIBRARIES})
```

<h3 id="sec5_5">CUDA</h3>

```cmake
find_package(CUDA REQUIRED)

# .cuファイルにincludeさせるときはtarget_include_directoriesではダメっぽい
# include_directories(${EIGEN3_INCLUDE_DIRS})

cuda_add_executable(main main.cpp kernel.cu)
```

<h3 id="sec5_6">OpenMP</h3>

```cmake
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
```

<h3 id="sec5_7">Threads</h3>

```cmake
find_package(Threads REQUIRED)
target_link_libraries(main PRIVATE Threads::Threads)
```

<h2 id="sec_cpp">C++</h2>

<h3 id="sec_cpp_1">Stopwatch</h3>

```cpp
#include <chrono>
#include <iostream>

int main() {
  std::chrono::system_clock::time_point start, end;
  start = std::chrono::system_clock::now();

  int cnt = 0;
  for (int i = 0; i < 1e8; i++) { cnt++; }

  end = std::chrono::system_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << elapsed << std::endl;
}
```

<h3 id="sec_cpp_2">CSV Reader</h3>

```cpp
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

std::vector<std::string> split(std::string& input, char delimiter) {
    std::istringstream stream(input);
    std::string val;
    std::vector<std::string> result;
    while (std::getline(stream, val, delimiter)) {
        result.push_back(val);
    }
    return result;
}

int main() {
    std::ifstream ifs("data.csv");

    std::string line;
    while (std::getline(ifs, line)) {
        std::vector<std::string> row = split(line, ',');
        for (int i = 0; i < row.size(); i++) {
            std::cout << row[i] << ", ";
        }
        std::cout << std::endl;
    }
}
```

<h3 id="sec_cpp_opencv">OpenCV Standard Operations</h3>

```cpp
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

int main() {
  // Initialize with values
  cv::Mat mat_cv = (cv::Mat_<double>(3, 3) << 2600, 0, 960, 0, 2600, 540, 0, 0, 1);
  // Convert OpenCV <-> Eigen
  Eigen::Matrix3d mat_eigen;
  cv2eigen(mat_cv, mat_eigen);
  eigen2cv(mat_eigen, mat_cv);

  // Initialize cv::Mat
  constexpr int HEIGHT = 1080;
  constexpr int WIDTH = 1920;
  cv::Mat img(HEIGHT, WIDTH, CV_8UC3, cv::Scalar(255, 255, 255));

  // Access to pixel
  int x = 0;
  int y = 0;
  img.at<cv::Vec3b>(y, x) = 0;
}
```

<h3 id="sec_cpp_filestorage">OpenCV File Storage</h3>

```cpp
{
  cv::FileStorage fs('filename.xml', cv::FileStorage::WRITE);
  fs << "intr" << intr;
}

{
  cv::FileStorage fs('filename.xml', cv::FileStorage::READ);
  fs["intr"] >> intr;
}
```

<h3 id="sec_cpp_fullscreen">OpenCV Fullscreen</h3>

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
  cv::namedWindow("Pattern", cv::WINDOW_NORMAL);
  cv::resizeWindow("Pattern", 1080, 1920);
  // 2枚目のディスプレイにフルスクリーン表示
  cv::moveWindow("Pattern", 0, 0);
  cv::setWindowProperty("Pattern", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

  cv::Mat img = cv::Mat::zeros(1080, 1920, CV_8U);

  cv::imshow("Pattern", img);
  cv::waitKey();
}
```

<h2 id="sec_py">Python</h2>

<h3 id="sec_py_argparse">argparse Template</h3>

```py
import argparse

parser = argparse.ArgumentParser(description='test app')

parser.add_argument('height', type=int, help='height[pix]')
parser.add_argument('-step', type=int, default=40, help='step')

args = parser.parse_args()
```

<h3 id="sec_py_1">CSV Reader</h3>

```py
import numpy as np

data = np.loadtxt("data.csv", delimiter=",")
```

<h3 id="sec_py_2">Simple XML Writer/Reader</h3>

```py
import sys
import numpy

print('<data>')
np.savetxt(sys.stdout.buffer, data, delimiter=',')
print('</data>')
```

```py
import sys
import re
from io import StringIO
import numpy as np

datastr = sys.stdin.read()
pattern = re.compile(r'<([^<>]+)>\s*(.+)\s*</\1>', re.MULTILINE | re.DOTALL)
valuelist = re.findall(pattern, datastr)
valuedict = dict(valuelist)

print("Keys : " + ', '.join([key for key, val in valuelist]))

data = np.loadtxt(StringIO(valuedict['data']), delimiter=',')
print(data)
```

<h3 id="sec_py_filestorage">OpenCV File Storage</h3>

```py
fs = cv2.FileStorage('filename.xml', cv2.FILE_STORAGE_WRITE)
fs.write('intr', intr)
fs.release()

fs = cv2.FileStorage('filename.xml', cv2.FileStorage_READ)
intr = fs.getNode('intr').mat()
fs.release()
```

<h3 id="sec_py_fullscreen">OpenCV Fullscreen</h3>

```py
import argparse
import cv2

parser = argparse.ArgumentParser(description='Display images with full screen')

parser.add_argument('--paths', nargs='+', help='images paths')
parser.add_argument('--xposs', type=int, nargs='+', help='x-positions')

args = parser.parse_args()

if len(args.paths) != len(args.xposs):
    print('Invalid inputs')

for i, (filename, xpos) in enumerate(zip(args.paths, args.xposs)):
    img = cv2.imread(filename)
    winname = 'img' + str(i+1)
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.moveWindow(winname, xpos, 0)
    cv2.setWindowProperty(
        winname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.imshow(winname, img)

cv2.waitKey()
```

<h3 id="sec_py_plotly">Plotly Simple Template</h3>

```py
import numpy as np

import plotly.offline as po
import plotly.graph_objs as go

# po.init_notebook_mode(connected=True)

xs = np.linspace(0, 100, 101)
ys1 = np.random.randn(len(xs))
ys2 = np.random.randn(len(xs))

trace1 = go.Scatter(x=xs, y=ys1, mode='markers')
trace2 = go.Scatter(x=xs, y=ys2, mode='lines+markers')

data = [trace1, trace2]
layout = dict(title='template of scatter')

fig = go.Figure(data=data, layout=layout)
po.plot(fig, filename='template_scatter.html')
# fig.show()
```

<h3 id="sec_py_3">Plotly 1D data</h3>

```py
import sys
import re
from io import StringIO
import numpy as np

datastr = sys.stdin.read()
pattern = re.compile(r'<([^<>]+)>\s*(.+)\s*</\1>', re.MULTILINE | re.DOTALL)
valuelist = re.findall(pattern, datastr)
valuedict = dict(valuelist)
print("Keys : " + ', '.join([key for key, val in valuelist]))


import plotly.offline as po
import plotly.graph_objs as go

data = []
for key, val in valuelist:
    arr = np.loadtxt(StringIO(val), delimiter=',')
    trace = go.Scatter(y=arr, name=key, mode='markers')
    data.append(trace)

layout = dict(title='test')

fig = go.Figure(data=data, layout=layout)
po.plot(fig, filename='test.html')
```

<h3 id="sec_py_plotly_figure">Plotly Figure</h3>

```py
fig = dict(
  data = [
    # Trace(グラフ化するデータを格納するオブジェクト)のリスト
    # グラフの種類に応じて
    #   Scatter/Scattergl/Bar/Box/Pie/Area/Heatmap
    #   Contour/Histogram/Histogram2D/Histogram2Dcontour
    #   Ohlc/Candlestick/Table/Scatter3D/Surface/Mesh3D
    # のいずれかが入る
    go.Scatter(
      x = [0, 1], y = [10, 20],  # データ(グラフの種類によっては文字列が入ったり、zがあったりする)
      mode = 'lines',
      line = dict(     # 線の書式
        width = 2,
      ),
      marker = dict(   # データ点の書式
        symbol = 'circle',
        size = 6,
      ),
      showlegend = True,  # 凡例にこのTraceを表示するかどうか
    ),
  ],
  layout = dict(
    font = dict(       # グローバルのフォント設定
      family = '"Open Sans", verdana, arial, sans-serif',
      size = 12,
      color = '#444',
    ),
    title = dict(
      text = '',       # グラフのタイトル
      font = dict(),   # タイトルのフォント設定
    ),
    width = 700,       # 全体のサイズ
    height = 450,
    autosize = True,   # HTMLで表示したときページに合わせてリサイズするかどうか
    margin = dict(     # グラフ領域の余白設定
      l = 80, r = 80, t = 100, b = 80,
      pad = 0,         # グラフから軸のラベルまでのpadding
      autoexpand = True,  # LegendやSidebarが被ったときに自動で余白を増やすかどうか
    ),
    xaxis = dict(      # 2Dグラフ用のx軸の設定
      title = dict(text = '', font = dict()),  # x軸のラベル
      type = '-',      # 'linear'、'log'、'date'などに設定可能
      autorange = True,
      range = [0, 1],
      scaleanchor = 'x1', scaleratio = 1,  # 他の軸とのスケールを固定したいときに使う
      tickmode = 'auto',  # 目盛りの刻み方(目盛り関連の設定項目は他にもいくつかあります)
    ),
    yaxis = dict(),    # 2Dグラフ用のy軸の設定、だいたいx軸と一緒
    scene = dict(      # 3Dグラフ用の設定
      camera = dict(),
      xaxis = dict(), yaxis = dict(), zaxis = dict()
    ),
    geo = dict(),      # グラフに地図を表示させるときの設定
    showlegend = True, # 凡例を表示するかどうか
    legend = dict(
      font = dict(),   # 凡例のフォント設定
      x = 1.02, xanchor = 'left',  # 凡例の表示場所の設定
      y = 1, yanchor = 'auto',
      bordercolor = '#444', borderwidth = 0,  # 凡例を囲む枠線の設定
    ),
    annotations = [],  # グラフ中に注釈を入れたいときに使う
    shapes = [],       # グラフ中に線や塗りつぶしを挿入したいときに使う
  ),
)
```

<h2 id="sec_simd">SIMD</h2>
<h3 id="sec_simd_check">Check supported architecture</h3>
Ubuntu
```sh
cat /proc/cpuinfo
```

mac
```sh
sysctl -a | grep cpu.features
sysctl -a | grep cpu.leaf7_features
```

<h3 id="sec_simd_sse">SSE</h3>

```c++
#include <stdio.h>
#include <xmmintrin.h>

int main(void) {
    __m128 a = {1.0f, 2.0f, 3.0f, 4.0f};
    __m128 b = {1.1f, 2.1f, 3.1f, 4.1f};
    float c[4];
    __m128 ps = _mm_add_ps(a, b); // add
    _mm_storeu_ps(c, ps); // store

    printf("  source: (%5.1f, %5.1f, %5.1f, %5.1f)\n",
            a[0], a[1], a[2], a[3]);
    printf("  dest. : (%5.1f, %5.1f, %5.1f, %5.1f)\n",
           b[0], b[1], b[2], b[3]);
    printf("  result: (%5.1f, %5.1f, %5.1f, %5.1f)\n",
           c[0], c[1], c[2], c[3]);
    return 0;
}
```

* `-msse4`のようにコンパイルオプションを付ける（デフォルトでenableになっているケースもあるっぽい）

<h3 id="sec_simd_avx">AVX</h3>

```c++
#include <stdio.h>
#include <immintrin.h>

int main(void) {
    __m256 a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    __m256 b = {1.1f, 2.1f, 3.1f, 4.1f, 5.1f, 6.1f, 7.1f, 8.1f};
    __m256 c;

    c = _mm256_add_ps(a, b);

    printf("  source: (%5.1f, %5.1f, %5.1f, %5.1f, %5.1f, %5.1f, %5.1f, %5.1f)\n",
            a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
    printf("  dest. : (%5.1f, %5.1f, %5.1f, %5.1f, %5.1f, %5.1f, %5.1f, %5.1f)\n",
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]);
    printf("  result: (%5.1f, %5.1f, %5.1f, %5.1f, %5.1f, %5.1f, %5.1f, %5.1f)\n",
           c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]);
    return 0;
}
```

* `-mavx2`のようにコンパイルオプションを付ける

<h2 id="sec_cuda">CUDA</h2>

<h3 id="sec_cuda_dim">Define Threads</h3>

1D

```cu
__global__ void test(..., int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N) {
  }
}

int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
test<<<blocksPerGrid, threadsPerBlock>>>(..., N);
```

2D

```cu
__global__ void test(..., int NX, int NY) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  if (x < NX && y < NY) {
  }
}

int tpb_x = 256
int tpb_y = 256
int bpg_x = (NX + tpb_x - 1) / tpb_x;
int bpg_y = (NY + tpb_y - 1) / tpb_y;
dim3 threadsPerBlock (tpb_x, tpb_y);
dim3 blocksPerGrid (bpg_x, bpg_y);
test<<<blocksPerGrid, threadsPerBlock>>>(..., NX, NY);
```

<h3 id="sec_cuda_stopwatch">Kernel Stopwatch</h3>

```cu
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);

// Kernel operation

cudaEventRecord(stop);
cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
std::cout << "Device processing time : " << milliseconds << " ms" << std::endl;
cudaEventDestroy(start);
cudaEventDestroy(stop);
```

<h2 id="sec_imgproc">Image Processing</h2>
<h3 id="sec_imgproc_chess">Generate checkerboard</h3>

```py
bsize = 60
img = np.kron(
    [[1,0]*int(1920/2/bsize), [0,1]*int(1920/2/bsize)]*int(1080/2/bsize),
    np.ones((bsize, bsize)))*255
```

<h2 id="sec_network">Network</h2>
<h3 id="sec_check_used_ips">Check used IPs</h3>

現在のarpテーブルを表示するだけだが、最近activeだったIPを調べるだけならこれで十分。

```sh
arp -a
```

<h3 id="sec_http_server">Temporal HTTP Server</h3>

```sh
python3 -m http.server 8000
```

```sh
npm install -g http-server
http-server -p 8000
```
