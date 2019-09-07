FROM conda/miniconda3
# Set environment
RUN apt-get -y update && \
    apt-get -y install gcc g++ curl apt-utils libgl1-mesa-dev qtbase5-dev libqt5opengl5-dev libassimp-dev patchelf cmake pkg-config git && \
    apt-get clean && \
    git clone https://github.com/openai/roboschool.git && \
    cd /roboschool && \
    bash -c ". ./exports.sh && ./install_boost.sh && ./install_bullet.sh && ./roboschool_compile_and_graft.sh" && \
    pip install -e /roboschool
RUN pip install click torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

RUN mkdir /app
WORKDIR /app
COPY . .