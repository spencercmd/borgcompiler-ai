FROM node:20-alpine

WORKDIR /app

# Install dependencies first (separate layer so npm install is cached
# as long as package.json hasn't changed).
COPY package*.json ./
RUN npm install

COPY . .

EXPOSE 5173

# --host 0.0.0.0 makes Vite listen on all interfaces inside the container,
# not just localhost — required for Docker port forwarding to reach it.
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
