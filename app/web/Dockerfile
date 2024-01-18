# Use an official Node.js image as the base image
FROM node:20-alpine

# Set the working directory inside the container
WORKDIR /app

# Copy package.json and package-lock.json to the working directory
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the entire application to the working directory
COPY . .

# Build the React app
RUN npm run build

# Set environment variables
ENV PORT 3000
ENV BACKEND_URL=http://localhost:8000

# Expose the port that the app will run on
EXPOSE 3000

# Command to run the application
CMD ["npm", "run", "start"]