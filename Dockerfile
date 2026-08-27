# The server app: browser-decoded frames in, gaussian splats out.
#
# There is nothing to install -- the server has no npm dependencies -- so this
# is a copy of the source onto a Node runtime. Reconstruction backends are NOT
# baked in: with no COLMAP on PATH the app runs its preview backend, which needs
# nothing. See docs/DEPLOY.md for the GPU image that adds real pose solving.
FROM node:22-alpine

WORKDIR /app

COPY package.json ./
COPY server ./server
COPY web ./web

# Bind on all interfaces (the app defaults to loopback, which a container hides)
# and keep the library on a volume so it survives replacing the container.
ENV NODE_ENV=production \
    HOST=0.0.0.0 \
    PORT=8787 \
    SPLAT_DATA_DIR=/data

RUN mkdir -p /data && chown -R node:node /data /app
USER node
VOLUME ["/data"]
EXPOSE 8787

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD node -e "fetch('http://127.0.0.1:'+(process.env.PORT||8787)+'/api/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"

CMD ["node", "server/index.js"]
