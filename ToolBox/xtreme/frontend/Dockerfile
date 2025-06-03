# syntax=docker/dockerfile:1
FROM node:16 as build
WORKDIR /build
COPY . .

# Install dependencies for main
WORKDIR /build/main
RUN npm install && \
    npm run build

# Install dependencies for image-tool
WORKDIR /build/image-tool
RUN npm install && \
    npm run build

# Install dependencies for pc-tool
WORKDIR /build/pc-tool
RUN npm install && \
    npm run build

# Install dependencies for text-tool
WORKDIR /build/text-tool
RUN npm install && \
    npm run build

FROM nginx:1.22
COPY --from=build /build/dist /usr/share/nginx/html