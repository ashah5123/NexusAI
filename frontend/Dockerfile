FROM node:22-slim

WORKDIR /app

COPY package.json ./
RUN npm install

COPY . .

EXPOSE 5173

# Phase 1 runs the Vite dev server directly (fast reload, no build step) rather than a
# production nginx image — revisit once there is a real deployment target.
CMD ["npm", "run", "dev"]
