# My Code Snippets

1. [Docker - Enable ctrl+p](#sec1)
1. [NeoVim Setup](#sec2)
    1. [Ubuntu 16.04](#sec2_1)
    1. [Ubuntu 18.04](#sec2_2)
    1. [Windows 10](#sec2_3)
1. [Git Setup](#sec3)
1. [Install Latest CMake into Ubuntu](#sec4)
1. [CMake Templates](#sec5)
    1. [Base](#sec5_1)
    1. [OpenCV](#sec5_2)
    1. [Eigen](#sec5_3)
    1. [Ceres](#sec5_4)
    1. [CUDA](#sec5_5)
1. [C++](#sec6)
    1. [Stopwatch](#sec6_1)
    1. [CSV Reader](#sec6_2)
1. [Python](#sec7)
    1. [CSV Reader](#sec7_1)
    1. [Simple XML Write/Reader](#sec7_2)
    1. [Plotly 1D data](#sec7_3)

<h2 id="sec1">Docker - Enable ctrl+p</h2>

`~/.docker/config.json`

```json
{
  "auths" : {
    "https://index.docker.io/v1/" : {
    }
  },
  "detachKeys" : "ctrl-\\"
}
```

<h2 id="sec2">NeoVim Setup</h2>

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
nnoremap <silent><C-e> :NERDTreeTabsToggle<CR>

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

<h2 id="sec4">Install Latest CMake into Ubuntu</h2>

```sh
apt install libncurses5-dev libcurl4-openssl-dev
wget https://github.com/Kitware/CMake/releases/download/v3.15.4/cmake-3.15.4.tar.gz
tar -xzf cmake-3.15.4.tar.gz
cd cmake-3.15.4
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

<h2 id="sec6">C++</h2>

<h3 id="sec6_1">C++ Stopwatch</h3>

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

<h3 id="sec6_2">CSV Reader</h3>

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

<h2 id="sec7">Python</h2>

<h3 id="sec7_1">CSV Reader</h3>

```py
import numpy as np

data = np.loadtxt("iris.csv", delimiter=",")
```

<h3 id="sec7_2">Simple XML Write/Reader</h3>

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

<h3 id="sec7_3">Plotly 1D data</h3>

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
