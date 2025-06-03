# syntax=docker/dockerfile:1
FROM maven:3.8-openjdk-11 as build
WORKDIR /build
ADD . .
RUN --mount=type=cache,target=/root/.m2 mvn package

FROM openjdk:11-jre
# 添加国内镜像源
RUN sed -i 's/deb.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list
# 分组安装软件包，提高成功率
RUN apt update && \
    apt install -y iputils-ping curl wget netcat && \
    apt install -y python3 python3-pip && \
    apt install -y git

# 配置 pip 使用国内镜像源
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 配置 git 以解决 TLS 问题
RUN git config --global http.sslVerify false && \
    git config --global http.postBuffer 524288000

# 安装 xtreme1-sdk，增加重试机制
RUN for i in $(seq 1 3); do \
    pip3 install --upgrade --force-reinstall git+https://github.com/xtreme1-io/xtreme1-sdk.git@6b53a73 && break || sleep 5; \
    done

WORKDIR /app
COPY --from=build /build/target/xtreme1-backend-0.9.1-SNAPSHOT.jar ./app.jar
RUN mkdir -p config
# RUN wget 'https://github.com/xtreme1-io/asset/raw/main/datasets/xtreme1-lidar-fusion-trial.zip' -O xtreme1-lidar-fusion-trial.zip
# RUN wget 'https://github.com/xtreme1-io/asset/raw/main/datasets/xtreme1-image-trial.zip' -O xtreme1-image-trial.zip
EXPOSE 8080
CMD ["java", "-jar", "app.jar"]